"""Analysis layer for the NeptuneOD observation pipeline.

The pipeline already produces two per-observation dataframes: the standard
observation frame (:mod:`orbitdet.preprocessing.schema`), which is the canonical
*ingest* contract, and the per-observation weights frame
(:mod:`orbitdet.observations.weighting.dataframe`), which carries residuals,
weights and grouping.  This package adds the missing third piece: an **access
layer** that joins the standard frame to the live
:class:`~tudatpy.estimation.observations.SingleObservationSet` objects through
the ``observation_set_index`` foreign key, and a set of pure **view** functions
that convert the canonical frame to other representations on demand.

Design rules
------------
* The derived frame keeps the **canonical** representation (radians, ``first`` /
  ``second`` components).  Conversions are views - pure functions returning
  copies - and are never written back.
* The standard frame is never mutated; only the access layer's derived frame
  gains columns.
* Weights are read, never recomputed.  :meth:`ObservationAccess.refresh`
  re-reads residuals and weights from the collection after estimation.

Example
-------
::

    from orbitdet.analysis import ObservationAccess

    access = ObservationAccess.build(collection, standard_frame)
    # ... run the estimation, which mutates the collection ...
    access.refresh()
"""

from .access import (
    JOINED_COLUMNS,
    OBSERVABLE_TYPE_COLUMN,
    REJECTED_COLUMN,
    SET_ID_COLUMN,
    SET_INDEX_COLUMN,
    ObservationAccess,
    SetKey,
)
from .basis import BasisTransform, block_diagonal_basis, identity_basis, scan_basis
from .views import (
    ARCSEC_PER_RADIAN,
    rotate,
    to_arcsec,
    to_mas,
    with_scan_frame,
)

__all__ = [
    # Access layer
    "ObservationAccess",
    "JOINED_COLUMNS",
    "SET_INDEX_COLUMN",
    "SET_ID_COLUMN",
    "OBSERVABLE_TYPE_COLUMN",
    "REJECTED_COLUMN",
    "SetKey",
    # Basis
    "BasisTransform",
    "scan_basis",
    "identity_basis",
    "block_diagonal_basis",
    # Views
    "to_arcsec",
    "to_mas",
    "rotate",
    "with_scan_frame",
    "ARCSEC_PER_RADIAN",
]
