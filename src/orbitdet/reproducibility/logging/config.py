# src/orbitdet/reproducibility/logging/config.py
"""Declarative configuration for the single logging pipeline.

The repository previously configured logging in several overlapping places
(Hydra's ``job_logging``, ``logging.basicConfig`` in ``runtime.py``, file-descriptor
capture, and Aim's terminal-log capture). That stack made ``cfg.logging.level`` a
no-op and sent every application error to stdout instead of stderr.

Everything now flows through one :func:`logging.config.dictConfig` call built
here. The design goals are:

* **stdout/stderr split** so Slurm ``.err`` files contain real errors;
* **never block the application thread** - in normal operation all sinks sit
  behind a single :class:`logging.handlers.QueueHandler` drained by a background
  listener;
* **file logs live in the Hydra run directory**, honouring ``hydra.job.chdir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility.logging.handlers import coerce_level

#: Format used by every sink. Matches the historical console format so existing
#: log-parsing scripts keep working.
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

#: Filename, next to the run log, for records at ERROR and above.
ERROR_LOG_NAME = "errors.log"

#: Name of the console handler that owns WARNING and above (the stderr stream).
STDERR_HANDLER_NAME = "console_err"

#: Name of the rotating file handler.
FILE_HANDLER_NAME = "file"

#: Name of the AimStack handler (no dictConfig entry; created by configure_logging).
AIM_HANDLER_NAME = "aim"


@dataclass
class AimSettings:
    """Settings for the AimStack logging sink.

    Grouped separately because Aim is an optional external sink that can fail
    independently of the local files and console.

    Attributes:
        enabled: Whether to forward records to Aim.
        level: Level for the Aim sink. Defaults to the console ``level``.
        max_warnings: How many Aim handler failures to report before going
            quiet.
        capture_terminal_logs: Whether Aim should also capture terminal output
            itself. Disabled by default: the Aim handler already carries the
            same content, and Aim's capture globally patches ``sys.stdout``,
            which previously conflicted with the other capture mechanisms.
    """

    enabled: bool = True
    level: str | int | None = None
    max_warnings: int = 3
    capture_terminal_logs: bool = False


@dataclass
class LoggingSettings:
    """Resolved logging settings.

    Built from the ``cfg.logging`` config group with :meth:`from_config`, which
    applies defaults for every optional field so a partial config is valid.

    Attributes:
        level: Level for the console streams and (later) the Aim sink.
        file_level: Level for the rotating file sink. Defaults to ``level``; set
            lower (e.g. ``DEBUG``) to keep a complete on-disk record while
            keeping the console quiet.
        tudatpy_logging_level: Level applied to the ``tudatpy`` logger.
        muted_loggers: Logger names forced to :data:`logging.WARNING`.
        filename: Base name of the run log file.
        max_bytes: Rotation size threshold in bytes.
        backup_count: Number of rotated files to keep.
        aim: Settings for the AimStack sink.
    """

    level: str | int = "INFO"
    file_level: str | int | None = None
    tudatpy_logging_level: str | int = "WARNING"
    muted_loggers: list[str] = field(default_factory=lambda: ["matplotlib"])
    filename: str = "run.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    aim: AimSettings = field(default_factory=AimSettings)

    @classmethod
    def from_config(cls, cfg: DictConfig | None) -> LoggingSettings:
        """Build settings from a Hydra config, tolerating missing keys.

        Args:
            cfg: The full Hydra config. When it has no ``logging`` section the
                dataclass defaults are used.

        Returns:
            The resolved settings.
        """
        section = OmegaConf.select(cfg, "logging") if cfg is not None else None
        if section is None:
            return cls()

        muted = section.get("muted_loggers", None)
        muted_loggers = [str(name) for name in muted] if muted is not None else None

        filename = section.get("filename", None)

        aim_section = section.get("aim", None)
        if aim_section is None:
            aim = AimSettings()
        else:
            aim = AimSettings(
                enabled=bool(aim_section.get("enabled", True)),
                level=aim_section.get("level", None),
                max_warnings=int(aim_section.get("max_warnings", 3)),
                capture_terminal_logs=bool(aim_section.get("capture_terminal_logs", False)),
            )

        return cls(
            level=section.get("level", cls.level),
            file_level=section.get("file_level", None),
            tudatpy_logging_level=section.get("tudatpy_logging_level", cls.tudatpy_logging_level),
            muted_loggers=muted_loggers if muted_loggers is not None else ["matplotlib"],
            filename=str(filename) if filename is not None else cls.filename,
            max_bytes=int(section.get("max_bytes", cls.max_bytes)),
            backup_count=int(section.get("backup_count", cls.backup_count)),
            aim=aim,
        )

    @property
    def effective_file_level(self) -> int:
        """Numeric level for the file sink, falling back to :attr:`level`."""
        return coerce_level(self.file_level if self.file_level is not None else self.level)

    @property
    def effective_aim_level(self) -> int:
        """Numeric level for the Aim sink, falling back to :attr:`level`."""
        return coerce_level(self.aim.level if self.aim.level is not None else self.level)

    @property
    def root_level(self) -> int:
        """Numeric root level - the most verbose of the enabled sinks."""
        levels = [coerce_level(self.level), self.effective_file_level]
        if self.aim.enabled:
            levels.append(self.effective_aim_level)
        return min(levels)


def build_sink_payload(
    settings: LoggingSettings,
    *,
    log_file: Path | None,
    error_file: Path | None = None,
) -> tuple[dict, list[str]]:
    """Build a ``dictConfig`` payload defining the sink handlers.

    The sinks are *defined* but not attached to the root logger, so the caller
    can either hand them to a :class:`~logging.handlers.QueueListener` (normal
    mode) or attach them directly (fallback mode). Retrieve the constructed
    instances afterwards with :func:`logging.getHandlerByName`, which also
    returns handlers that are not attached to any logger.

    Args:
        settings: Resolved logging settings.
        log_file: Destination for the rotating run log, or ``None`` to skip it.
        error_file: Destination for ERROR-and-above records, or ``None`` to skip
            it. Defaults to :data:`ERROR_LOG_NAME` beside ``log_file``.

    Returns:
        A ``(payload, sink_names)`` tuple. ``payload`` is ready for
        :func:`logging.config.dictConfig`.
    """
    if error_file is None and log_file is not None:
        error_file = log_file.with_name(ERROR_LOG_NAME)

    formatters = {
        "standard": {"format": LOG_FORMAT, "datefmt": LOG_DATEFMT},
    }
    filters = {"below_warning": {"()": "orbitdet.reproducibility.logging.handlers.MaxLevelFilter"}}

    handlers: dict = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
            "level": coerce_level(settings.level),
            "filters": ["below_warning"],
        },
        STDERR_HANDLER_NAME: {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
            "level": coerce_level("WARNING"),
        },
    }

    sink_names: list[str] = ["console", STDERR_HANDLER_NAME]

    if log_file is not None:
        handlers[FILE_HANDLER_NAME] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(log_file),
            "maxBytes": settings.max_bytes,
            "backupCount": settings.backup_count,
            "encoding": "utf-8",
            "level": settings.effective_file_level,
            "delay": True,
        }
        sink_names.append(FILE_HANDLER_NAME)

    if error_file is not None:
        handlers["error_file"] = {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": str(error_file),
            "encoding": "utf-8",
            "level": coerce_level("ERROR"),
            "delay": True,
        }
        sink_names.append("error_file")

    payload = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "loggers": _aux_loggers(settings),
        # Attached by the caller: a QueueHandler in normal mode, the sinks
        # themselves in fallback mode.
        "root": {"level": settings.root_level, "handlers": []},
    }
    return payload, sink_names


def _aux_loggers(settings: LoggingSettings) -> dict:
    """Return per-logger overrides for tudatpy and the muted loggers."""
    loggers: dict = {
        "tudatpy": {"level": coerce_level(settings.tudatpy_logging_level)},
    }
    for name in settings.muted_loggers:
        loggers[name] = {"level": coerce_level("WARNING")}
    return loggers
