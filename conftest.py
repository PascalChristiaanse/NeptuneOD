"""Shared pytest configuration for the NeptuneOD test suite.

The Gaia astrometry adapter (:mod:`orbitdet.data.gaia`) is a thin wrapper around
``tudatpy.data.gaia``.  That module only exists in the custom tudatpy build used
for development; the tudatpy release published on GitHub does not ship it yet.
Because :mod:`orbitdet.data` imports the adapter eagerly, every test module that
touches ``orbitdet.data`` or ``orbitdet.observations`` fails at collection time
on CI with ``ModuleNotFoundError: No module named 'tudatpy.data.gaia'``.

To keep the suite collectable in environments without the Gaia interface, this
conftest installs a stub ``tudatpy.data.gaia`` module when the real one is
missing.  The stub only satisfies the import: any attempt to actually *use* the
Gaia functionality raises a clear :class:`RuntimeError` instead of silently
returning bogus data.  When the real module is available (the development
environment) nothing is patched, so this file can be deleted once tudatpy ships
``tudatpy.data.gaia`` in a released build.
"""

from __future__ import annotations

import importlib
import sys
import types

_GAIA_PACKAGE = "tudatpy.data.gaia"
_GAIA_IMPL_MODULE = "tudatpy.data.gaia.gaia"
_ASTROMETRY_CATALOG = "gaiafpr.sso_observation"

_UNAVAILABLE_MESSAGE = (
    "tudatpy.data.gaia is not available in this environment. The Gaia astrometry "
    "interface requires a tudatpy build that ships tudatpy.data.gaia; the release "
    "published on GitHub does not include it yet. Install the development build of "
    "tudatpy to use the Gaia functionality."
)


def _unavailable(*_args: object, **_kwargs: object) -> None:
    """Raise a descriptive error for any use of the stubbed Gaia interface."""
    raise RuntimeError(_UNAVAILABLE_MESSAGE)


class GaiaAstrometry:
    """Import-compatible stand-in for ``tudatpy.data.gaia.GaiaAstrometry``.

    Instantiating the class or calling any of its public constructors raises a
    :class:`RuntimeError` explaining that the real interface is unavailable.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        _unavailable()

    load_from_astroquery = staticmethod(_unavailable)
    load_from_astroquery_for_source_ids = staticmethod(_unavailable)
    load_from_local_archive = staticmethod(_unavailable)
    load_from_local_archive_for_source_ids = staticmethod(_unavailable)
    _prepare_table = staticmethod(_unavailable)


def _ensure_package(name: str) -> types.ModuleType:
    """Return the module ``name``, creating an empty package if it is missing."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]  # mark as a package
    sys.modules[name] = module

    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        setattr(_ensure_package(parent_name), child_name, module)
    return module


def _install_gaia_stub() -> None:
    """Register a stub ``tudatpy.data.gaia`` if the real module cannot be imported."""
    try:
        importlib.import_module(_GAIA_PACKAGE)
    except ModuleNotFoundError:
        pass
    else:
        return  # Real interface available; leave it untouched.

    parent = _ensure_package("tudatpy.data")

    implementation = types.ModuleType(_GAIA_IMPL_MODULE)
    implementation.GaiaAstrometry = GaiaAstrometry
    implementation._ASTROMETRY_CATALOG = _ASTROMETRY_CATALOG

    package = types.ModuleType(_GAIA_PACKAGE)
    package.__path__ = []  # type: ignore[attr-defined]  # mark as a package
    package.GaiaAstrometry = GaiaAstrometry
    package._ASTROMETRY_CATALOG = _ASTROMETRY_CATALOG
    package.gaia = implementation

    sys.modules[_GAIA_IMPL_MODULE] = implementation
    sys.modules[_GAIA_PACKAGE] = package
    parent.gaia = package


_install_gaia_stub()
