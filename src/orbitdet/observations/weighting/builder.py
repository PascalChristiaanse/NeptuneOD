"""Build weight strategies from configuration descriptors.

A *descriptor* is either:

- a plain strategy name (legacy), e.g. ``"id_v2"``; or
- a mapping with a ``type`` key plus constructor kwargs, e.g.::

      {type: layered, combine: geometric, layers: [...]}

Descriptors may nest: a ``router`` route's ``source`` is itself a descriptor.
Keeping this logic in one place lets both :class:`WeightEngine` and the
``router`` strategy build sub-strategies identically.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from .registry import get_strategy_class


def to_plain(value: Any) -> Any:
    """Convert an OmegaConf node to plain Python containers (recursively)."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_strategy(spec: Any, grouping: Any = None):
    """Instantiate a weight strategy from a descriptor.

    Parameters
    ----------
    spec : str | Mapping
        Either a strategy name (e.g. ``"id_v2"``) or a mapping with a ``type``
        key plus constructor kwargs.
    grouping : Any, optional
        The grouping config node.  Passed to strategies that expose a
        ``set_grouping`` hook (e.g. the ``layered`` strategy).

    Returns
    -------
    WeightStrategy
        The constructed strategy instance.
    """
    if isinstance(spec, str):
        name = spec
        params: dict[str, Any] = {}
    elif hasattr(spec, "items"):
        plain = to_plain(spec)
        if "type" not in plain:
            raise ValueError(f"Strategy descriptor must have a 'type' key, got keys {list(plain)}.")
        name = str(plain["type"])
        params = {k: v for k, v in plain.items() if k != "type"}
    else:
        raise TypeError(
            f"Strategy descriptor must be a name or a mapping, got {type(spec).__name__}."
        )

    strategy_cls = get_strategy_class(name)
    strategy = strategy_cls(**params)
    if hasattr(strategy, "set_grouping"):
        strategy.set_grouping(grouping)
    return strategy
