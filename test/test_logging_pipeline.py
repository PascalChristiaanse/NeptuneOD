"""Tests for the single logging pipeline (orbitdet.reproducibility.logging)."""

import logging
import queue
import time

import pytest
from omegaconf import OmegaConf

from orbitdet.reproducibility import logging as od_logging
from orbitdet.reproducibility.logging import config as od_config
from orbitdet.reproducibility.logging.handlers import (
    AimLogHandler,
    MaxLevelFilter,
    coerce_level,
)
from orbitdet.reproducibility.logging.listener import BoundedQueueListener


@pytest.fixture(autouse=True)
def clean_logging():
    """Restore a pristine logging state around every test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    yield

    od_logging.shutdown_logging()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


# ---------------------------------------------------------------------------
# coerce_level
# ---------------------------------------------------------------------------


def test_coerce_level_accepts_names_case_insensitively():
    assert coerce_level("info") == logging.INFO
    assert coerce_level("WARNING") == logging.WARNING
    assert coerce_level("debug") == logging.DEBUG


def test_coerce_level_accepts_numbers():
    assert coerce_level(10) == logging.DEBUG
    assert coerce_level(40) == logging.ERROR


def test_coerce_level_falls_back_to_info_for_unknown_names():
    """A typo must not silently disable logging."""
    assert coerce_level("NOT_A_LEVEL") == logging.INFO
    assert coerce_level(None) == logging.INFO


def test_coerce_level_rejects_bool():
    """bool is an int subclass but is never a valid level."""
    assert coerce_level(True) == logging.INFO


# ---------------------------------------------------------------------------
# MaxLevelFilter
# ---------------------------------------------------------------------------


def _record(level: int) -> logging.LogRecord:
    return logging.LogRecord("t", level, __file__, 1, "msg", None, None)


def test_max_level_filter_rejects_warning_and_above():
    f = MaxLevelFilter(logging.WARNING)
    assert f.filter(_record(logging.DEBUG)) is True
    assert f.filter(_record(logging.INFO)) is True
    assert f.filter(_record(logging.WARNING)) is False
    assert f.filter(_record(logging.ERROR)) is False
    assert f.filter(_record(logging.CRITICAL)) is False


# ---------------------------------------------------------------------------
# LoggingSettings
# ---------------------------------------------------------------------------


def test_settings_from_config_reads_every_field():
    cfg = OmegaConf.create(
        {
            "logging": {
                "level": "WARNING",
                "file_level": "DEBUG",
                "tudatpy_logging_level": "ERROR",
                "muted_loggers": ["a", "b"],
                "filename": "custom.log",
                "max_bytes": 1234,
                "backup_count": 9,
            }
        }
    )
    s = od_config.LoggingSettings.from_config(cfg)
    assert s.level == "WARNING"
    assert s.file_level == "DEBUG"
    assert s.tudatpy_logging_level == "ERROR"
    assert s.muted_loggers == ["a", "b"]
    assert s.filename == "custom.log"
    assert s.max_bytes == 1234
    assert s.backup_count == 9


def test_settings_from_config_applies_defaults_for_partial_config():
    cfg = OmegaConf.create({"logging": {"level": "WARNING"}})
    s = od_config.LoggingSettings.from_config(cfg)
    assert s.level == "WARNING"
    assert s.file_level is None
    assert s.filename == "run.log"
    assert s.muted_loggers == ["matplotlib"]


def test_settings_from_config_without_section_uses_defaults():
    assert od_config.LoggingSettings.from_config(OmegaConf.create({})).level == "INFO"
    assert od_config.LoggingSettings.from_config(None).level == "INFO"


def test_documented_config_matches_defaults():
    """The shipped conf/logging/default.yaml must parse into the defaults."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "conf" / "logging" / "default.yaml"
    raw = yaml.safe_load(path.read_text())["logging"]

    # `null` is the documented way to mean "inherit `level`".
    assert raw["file_level"] is None
    s = od_config.LoggingSettings.from_config(OmegaConf.create({"logging": raw}))
    assert s.effective_file_level == logging.INFO


def test_file_level_defaults_to_console_level():
    s = od_config.LoggingSettings(level="WARNING")
    assert s.effective_file_level == logging.WARNING
    assert s.root_level == logging.WARNING


def test_root_level_is_most_verbose_of_console_and_file():
    s = od_config.LoggingSettings(level="WARNING", file_level="DEBUG")
    assert s.root_level == logging.DEBUG


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_payload_split_console_streams(tmp_path):
    s = od_config.LoggingSettings(level="INFO")
    payload, sinks = od_config.build_sink_payload(s, log_file=tmp_path / "run.log")

    assert payload["handlers"]["console"]["stream"] == "ext://sys.stdout"
    assert payload["handlers"]["console_err"]["stream"] == "ext://sys.stderr"
    assert payload["handlers"]["console"]["filters"] == ["below_warning"]
    assert "filters" not in payload["handlers"]["console_err"]

    # Root is intentionally unattached; configure_logging wires it up.
    assert payload["root"]["handlers"] == []
    assert sinks == ["console", "console_err", "file", "error_file"]


def test_payload_omits_file_handlers_without_log_file():
    payload, sinks = od_config.build_sink_payload(od_config.LoggingSettings(), log_file=None)
    assert sinks == ["console", "console_err"]
    assert "file" not in payload["handlers"]
    assert "error_file" not in payload["handlers"]


def test_payload_mutes_configured_loggers():
    s = od_config.LoggingSettings(muted_loggers=["noisy", "also_noisy"])
    payload, _ = od_config.build_sink_payload(s, log_file=None)
    assert payload["loggers"]["noisy"]["level"] == logging.WARNING
    assert payload["loggers"]["also_noisy"]["level"] == logging.WARNING
    assert payload["loggers"]["tudatpy"]["level"] == logging.WARNING


# ---------------------------------------------------------------------------
# BoundedQueueListener
# ---------------------------------------------------------------------------


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def test_listener_delivers_records_and_reports_drained():
    sink = _CollectingHandler()
    q: queue.Queue = queue.Queue()
    listener = BoundedQueueListener(q, sink)
    listener.start()
    try:
        logging.getLogger("probe").addHandler(logging.handlers.QueueHandler(q))
        logging.getLogger("probe").warning("hello")
    finally:
        listener.stop()

    assert sink.records == ["hello"]
    assert listener.drained is True


def test_listener_respects_sink_levels():
    """A sink's own level must be honoured - this is the stderr split guarantee."""
    low = _CollectingHandler()
    low.setLevel(logging.DEBUG)
    high = _CollectingHandler()
    high.setLevel(logging.WARNING)

    q: queue.Queue = queue.Queue()
    listener = BoundedQueueListener(q, low, high)
    listener.start()
    try:
        logger = logging.getLogger("probe2")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.handlers.QueueHandler(q))
        logger.info("info")
        logger.error("error")
    finally:
        listener.stop()

    assert low.records == ["info", "error"]
    assert high.records == ["error"]


def test_listener_stop_does_not_hang_on_a_wedged_sink():
    """A sink that blocks forever must not be able to hang shutdown."""
    import threading

    # A sink that parks until released. This stands in for a network share that
    # has gone away or a stopped Aim server.
    release = threading.Event()

    class WedgedHandler(logging.Handler):
        def emit(self, record):
            release.wait(60)

    wedge = WedgedHandler()
    q: queue.Queue = queue.Queue()
    listener = BoundedQueueListener(q, wedge, join_timeout=0.2)
    listener.start()

    logger = logging.getLogger("probe3")
    logger.addHandler(logging.handlers.QueueHandler(q))
    logger.warning("this will wedge the sink")

    start = time.monotonic()
    listener.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, "stop() must be bounded by join_timeout"
    assert listener.drained is False

    # Release the abandoned thread so it does not outlive the test.
    release.set()


def test_listener_stop_is_idempotent_and_safe_when_never_started():
    listener = BoundedQueueListener(queue.Queue(), _CollectingHandler(), join_timeout=0.1)
    listener.stop()  # never started
    listener.stop()  # idempotent


# ---------------------------------------------------------------------------
# configure_logging / shutdown_logging
# ---------------------------------------------------------------------------


def test_configure_logging_writes_split_streams_and_file(tmp_path):
    cfg = OmegaConf.create({"logging": {"level": "INFO", "muted_loggers": []}})
    run_dir = tmp_path / "run"

    od_logging.configure_logging(cfg, run_dir=run_dir, use_queue=True)

    assert od_logging.is_configured()
    logger = logging.getLogger("orbitdet.test")
    logger.info("an info line")
    logger.error("an error line")

    od_logging.shutdown_logging()

    log_text = (run_dir / "run.log").read_text()
    assert "an info line" in log_text
    assert "an error line" in log_text

    errors_text = (run_dir / "errors.log").read_text()
    assert "an error line" in errors_text
    assert "an info line" not in errors_text


def test_configure_logging_is_idempotent(tmp_path):
    cfg = OmegaConf.create({"logging": {"level": "INFO", "muted_loggers": []}})
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)
    listener_before = od_logging.get_listener()

    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)

    assert od_logging.get_listener() == listener_before
    root = logging.getLogger()
    assert len([h for h in root.handlers if h.name == "file" or not h.name]) >= 1


def test_reconfiguring_for_a_new_run_dir_does_not_leak_into_the_old_file(tmp_path):
    """A second job in the same process must not write into the first's log.

    Hydra's joblib launcher can run several jobs in one interpreter; sharing a
    process must not silently merge their run logs.
    """
    cfg = OmegaConf.create({"logging": {"level": "INFO", "muted_loggers": []}})
    first, second = tmp_path / "first", tmp_path / "second"

    od_logging.configure_logging(cfg, run_dir=first, use_queue=True)
    logging.getLogger("orbitdet.job1").info("first-job-line")

    od_logging.configure_logging(cfg, run_dir=second, use_queue=True)
    logging.getLogger("orbitdet.job2").info("second-job-line")

    od_logging.shutdown_logging()

    assert "first-job-line" in (first / "run.log").read_text()
    assert "second-job-line" not in (first / "run.log").read_text()
    assert "second-job-line" in (second / "run.log").read_text()


def test_configure_logging_honours_level(tmp_path):
    cfg = OmegaConf.create({"logging": {"level": "WARNING", "muted_loggers": []}})
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=True)

    logging.getLogger("orbitdet.quiet").info("should be dropped")
    logging.getLogger("orbitdet.loud").warning("should appear")

    od_logging.shutdown_logging()

    log_text = (tmp_path / "run.log").read_text()
    assert "should appear" in log_text
    assert "should be dropped" not in log_text


def test_configure_logging_without_run_dir_is_console_only():
    cfg = OmegaConf.create({"logging": {"level": "INFO", "muted_loggers": []}})
    od_logging.configure_logging(cfg, run_dir=None, use_queue=True)

    assert od_logging.is_configured()
    assert logging.getHandlerByName("file") is None

    logging.getLogger("orbitdet.console_only").info("console only")
    od_logging.shutdown_logging()


def test_direct_mode_attaches_sinks_to_root(tmp_path):
    cfg = OmegaConf.create({"logging": {"level": "INFO", "muted_loggers": []}})
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)

    root = logging.getLogger()
    assert od_logging.get_listener() is None
    # No queued indirection in fallback mode.
    assert not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers)

    logging.getLogger("orbitdet.direct").error("direct error")
    od_logging.shutdown_logging()

    assert "direct error" in (tmp_path / "run.log").read_text()


def test_shutdown_logging_is_safe_when_not_configured():
    od_logging.shutdown_logging()
    od_logging.shutdown_logging()


# ---------------------------------------------------------------------------
# AimLogHandler
# ---------------------------------------------------------------------------


class _FakeAimRun:
    """Records the objects an AimLogHandler would track."""

    def __init__(self):
        self.tracked = []

    def track(self, obj, name=None, **kwargs):
        self.tracked.append((name, obj))


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    name: str = "orbitdet.fake",
    pathname: str = "/src/module.py",
    lineno: int = 17,
) -> logging.LogRecord:
    return logging.LogRecord(name, level, pathname, lineno, msg, None, None)


def test_aim_handler_forwards_records_with_real_call_site():
    run = _FakeAimRun()
    handler = AimLogHandler(run=run, level=logging.INFO)

    handler.emit(_make_record("a message", logging.WARNING))

    assert len(run.tracked) == 1
    sequence_name, obj = run.tracked[0]
    assert sequence_name == AimLogHandler.SEQUENCE_NAME
    assert obj.message == "a message"
    assert obj.level == logging.WARNING
    # The application's call site must be preserved, not the handler's frame.
    assert obj.storage["__logger_info"] == ("/src/module.py", 17)


def test_aim_handler_interpolates_args():
    run = _FakeAimRun()
    handler = AimLogHandler(run=run, level=logging.INFO)

    record = logging.LogRecord("orbitdet.fake", logging.INFO, "/f.py", 3, "value=%s", ("x",), None)
    handler.emit(record)

    assert run.tracked[0][1].message == "value=x"


def test_aim_handler_drops_records_when_detached():
    run = _FakeAimRun()
    handler = AimLogHandler(run=run, level=logging.INFO)

    handler.attach_run(None)
    handler.emit(_make_record("dropped"))

    assert run.tracked == []
    assert handler.failure_count == 0


def test_aim_handler_attach_run_switches_target():
    first, second = _FakeAimRun(), _FakeAimRun()
    handler = AimLogHandler(run=first, level=logging.INFO)

    handler.emit(_make_record("to first"))
    handler.attach_run(second)
    handler.emit(_make_record("to second"))

    assert [o.message for _, o in first.tracked] == ["to first"]
    assert [o.message for _, o in second.tracked] == ["to second"]
    assert handler.run is second


def test_aim_handler_tolerates_failures_and_warns_at_most_max_warnings(capsys):
    class BrokenRun:
        def track(self, *args, **kwargs):
            raise RuntimeError("aim server unreachable")

    handler = AimLogHandler(run=BrokenRun(), level=logging.INFO, max_warnings=2)

    for i in range(10):
        handler.emit(_make_record(f"message {i}", logging.WARNING, name=f"logger{i}"))

    assert handler.failure_count == 10

    err = capsys.readouterr().err
    # No traceback may escape a handler, and warnings must be bounded.
    assert err.count("[logging] AimLogHandler failed") <= 2
    assert "RuntimeError" in err


def test_aim_handler_suppresses_repeat_warning_for_same_logger_and_level(capsys):
    class BrokenRun:
        def track(self, *args, **kwargs):
            raise RuntimeError("down")

    handler = AimLogHandler(run=BrokenRun(), level=logging.INFO, max_warnings=5)
    for _ in range(5):
        handler.emit(_make_record("same", logging.ERROR, name="same.logger"))

    assert capsys.readouterr().err.count("[logging] AimLogHandler failed") == 1


def test_aim_handler_obeys_level_filter():
    run = _FakeAimRun()
    handler = AimLogHandler(run=run, level=logging.WARNING)

    # Level filtering is applied by Logger.callHandlers, so attach the handler
    # to a logger rather than calling emit() directly.
    logger = logging.getLogger("orbitdet.aim_level")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        logger.info("info")
        logger.warning("warning")
    finally:
        logger.handlers[:] = []

    assert [o.message for _, o in run.tracked] == ["warning"]


def test_configure_logging_creates_detached_aim_handler(tmp_path):
    cfg = OmegaConf.create(
        {"logging": {"level": "INFO", "muted_loggers": [], "aim": {"enabled": True}}}
    )
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)

    handler = od_logging.get_aim_handler()
    assert handler is not None
    assert handler.run is None  # attached later by initialize()

    od_logging.shutdown_logging()
    assert od_logging.get_aim_handler() is None


def test_configure_logging_omits_aim_handler_when_disabled(tmp_path):
    cfg = OmegaConf.create(
        {"logging": {"level": "INFO", "muted_loggers": [], "aim": {"enabled": False}}}
    )
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)

    assert od_logging.get_aim_handler() is None


def test_attach_aim_run_points_the_handler_at_the_run(tmp_path):
    cfg = OmegaConf.create(
        {"logging": {"level": "INFO", "muted_loggers": [], "aim": {"enabled": True}}}
    )
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=True)

    run = _FakeAimRun()
    od_logging.attach_aim_run(run)

    assert od_logging.get_aim_handler().run is run

    logging.getLogger("orbitdet.aim").warning("routed to aim")
    od_logging.shutdown_logging()

    assert [o.message for _, o in run.tracked] == ["routed to aim"]


def test_attach_aim_run_is_safe_without_an_aim_handler(tmp_path):
    cfg = OmegaConf.create(
        {"logging": {"level": "INFO", "muted_loggers": [], "aim": {"enabled": False}}}
    )
    od_logging.configure_logging(cfg, run_dir=tmp_path, use_queue=False)

    # Must not raise.
    od_logging.attach_aim_run(_FakeAimRun())


def test_aim_terminal_capture_is_disabled_by_default():
    """Aim must not monkey-patch sys.stdout/sys.stderr.

    Aim's ``capture_terminal_logs`` replaces the stream ``write`` methods at
    class level, duplicating every line and conflicting with the logging
    pipeline. The AimLogHandler replaces it, so capture stays off.
    """
    assert od_config.AimSettings().capture_terminal_logs is False
    assert (
        od_config.LoggingSettings.from_config(OmegaConf.create({})).aim.capture_terminal_logs
        is False
    )

    # A config may re-enable it explicitly.
    cfg = OmegaConf.create({"logging": {"aim": {"capture_terminal_logs": True}}})
    assert od_config.LoggingSettings.from_config(cfg).aim.capture_terminal_logs is True


def test_shipped_config_disables_aim_terminal_capture():
    """conf/logging/default.yaml must not re-enable Aim's stream patching."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "conf" / "logging" / "default.yaml"
    raw = yaml.safe_load(path.read_text())["logging"]

    assert raw["aim"]["capture_terminal_logs"] is False
