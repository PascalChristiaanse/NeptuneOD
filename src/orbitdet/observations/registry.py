"""Registry for observation dataset factories."""

import logging
from collections.abc import Callable, Sequence
from typing import TypeVar

from .configs import ObservationDatasetConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic return type for factories

#: Separator between the two halves of a dispatch key.
DISPATCH_SEPARATOR = "|"

# Global factory registry: maps dataset type string to factory callable
_FACTORY_REGISTRY: dict[str, Callable[[ObservationDatasetConfig], T]] = {}


def dispatch_key(observation_type: str, coordinate_kind: str) -> str:
    """Build the normalised key that selects an observation factory.

    One NSDB ``type`` string encodes three independent facts at once - whether
    the measurement is absolute or relative, which coordinates it holds, and
    which receptor recorded it - and the observation *model* depends on only the
    first two.  The receptor is provenance, and already has its own column in the
    standard frame.

    Splitting the key lets one factory serve every dataset that shares a model,
    which is what makes the corpus tractable: the fused strings number 15+ with
    typos among them (``photograpihc``, ``no inf``), while
    ``(observation_type, coordinate_kind)`` has four combinations that occur.

    Args:
        observation_type: ``"absolute"`` or ``"relative"``.
        coordinate_kind: ``"ra_dec"``, ``"cartesian_xy"`` or
            ``"position_angle_separation"``.

    Returns:
        The key, e.g. ``"absolute|ra_dec"``.

    Examples:
        >>> dispatch_key("relative", "cartesian_xy")
        'relative|cartesian_xy'
    """
    return f"{observation_type}{DISPATCH_SEPARATOR}{coordinate_kind}"


def register_dataset_factory(
    dataset_type: str,
    *,
    aliases: Sequence[str] = (),
) -> Callable:
    """Decorator to register a dataset factory function.

    Args:
        dataset_type: The canonical type string (e.g. ``"absolute|ra_dec"``, or a
            legacy fused name such as ``"ground_ccd"``).
        aliases: Additional names that resolve to the same factory.  Used to keep
            the existing fused type strings in ``conf/datasets/generated``
            working while the canonical name moves to a dispatch key, so no
            committed config has to be rewritten for the refactor.

    Returns:
        Decorator function that registers the wrapped factory.

    Raises:
        ValueError: If any name is already registered.  Checked for every name
            *before* anything is registered, so a clash leaves the registry
            untouched rather than half-updated.

    Example:
        @register_dataset_factory('ground_ccd')
        def create_ground_ccd_dataset(cfg: GroundCCDConfig):
            ...
    """

    def decorator(factory_func: Callable) -> Callable:
        names = (dataset_type, *aliases)
        clashes = [name for name in names if name in _FACTORY_REGISTRY]
        if clashes:
            raise ValueError(
                f"Dataset factory for type '{clashes[0]}' is already registered. "
                f"Duplicate registration of {factory_func.__module__}.{factory_func.__name__} "
                f"conflicts with existing registration."
            )
        for name in names:
            _FACTORY_REGISTRY[name] = factory_func
        logger.debug(
            "Registered observation factory for %s (%d name(s))",
            dataset_type,
            len(names),
        )
        return factory_func

    return decorator


def get_factory(dataset_type: str) -> Callable:
    """Retrieve a registered factory by dataset type.

    Args:
        dataset_type: The dataset type string to look up.

    Returns:
        The registered factory callable.

    Raises:
        ValueError: If dataset_type is not registered.
    """
    if dataset_type not in _FACTORY_REGISTRY:
        available = ", ".join(sorted(_FACTORY_REGISTRY.keys()))
        raise ValueError(
            f"No factory registered for dataset type '{dataset_type}'. Available types: {available}"
        )
    return _FACTORY_REGISTRY[dataset_type]


def resolve_factory(dataset_cfg) -> Callable:
    """Find the factory for a dataset config, preferring normalised dispatch.

    The standard frame carries ``observation_type`` and ``coordinate_kind`` as
    columns, so a config that has been through preprocessing dispatches on those
    two values and never touches the fused type string.  A config that has not
    (every one committed under ``conf/datasets/generated``, plus the non-NSDB
    modalities) falls back to its ``type``, which the registry still holds as an
    alias.

    Args:
        dataset_cfg: A dataset config, normally a Hydra ``DictConfig``.

    Returns:
        The registered factory.

    Raises:
        ValueError: If neither the dispatch key nor ``type`` resolves.
    """
    from omegaconf import OmegaConf

    observation_type = OmegaConf.select(dataset_cfg, "observation_type")
    coordinate_kind = OmegaConf.select(dataset_cfg, "coordinate_kind")

    if observation_type and coordinate_kind:
        key = dispatch_key(str(observation_type), str(coordinate_kind))
        if key in _FACTORY_REGISTRY:
            return _FACTORY_REGISTRY[key]
        logger.debug(
            "No factory registered for dispatch key %r; falling back to the "
            "dataset's 'type' field.",
            key,
        )

    dataset_type = OmegaConf.select(dataset_cfg, "type")
    if not dataset_type:
        raise ValueError(
            "Dataset config must have either a 'type' field, or both "
            "'observation_type' and 'coordinate_kind'."
        )
    return get_factory(str(dataset_type))


def list_registered_types() -> list[str]:
    """Return a sorted list of all registered dataset types.

    Returns:
        List of registered dataset type strings.
    """
    return sorted(_FACTORY_REGISTRY.keys())
