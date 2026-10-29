"""Surface-reservoir geometry and exchange fluxes.

The six kernels of the BFS model, compiled with numba. References are to the
USGS BFS manual (Konrad, 2022, SIR 2022-5114); ``notes/geometry_reconciliation.md``
documents each comparison in detail.

Geometry (manual p. 4-5, fig. 1): the basin is an "open book" of two
symmetric hillslopes, each ``ws = Wb/2`` wide and ``Lb`` long, draining to a
central channel. The surface reservoir's upper boundary rises from the
channel at slope α and its lower boundary is level with the channel. The
saturated thickness at the section where the water surface meets the land
surface is ``Zs``, at lateral distance ``Xs = Zs/α`` from the channel (eq. 3):
land between the channel and Xs is saturated, land beyond it is not.

Symbols used as argument names (all lengths in m, flows in m³/day):

- ``lb``: basin length Lb
- ``ws``: hillslope width Wb/2
- ``alpha``: surface gradient α
- ``por``: drainable porosity POR
- ``zs``: surface saturated thickness Zs
- ``ks``, ``kz``: conductivities Ks, Kz [m/day]
- ``xb``: base-reservoir intersect distance Xb
- ``impulse``: impulse depth I [m/day]

The arithmetic is written exactly as in pybfs 0.2.0 (``pybfs/legacy``), so
results are bit-identical, including the known inconsistency between
`surface_storage_from_thickness` and `surface_thickness_from_storage`
(suspected issue M10).
"""

import math

from numba import njit


@njit(cache=True)
def surface_storage_from_thickness(lb, alpha, ws, por, zs):
    """Surface storage Ss for saturated thickness Zs [m³].

    ``Ss = Lb·POR·(2·ws·Zs − Zs²/α)``: two hillslopes, each saturated to Zs
    beyond Xs and to the land surface between the channel and Xs.

    Manual: eq. 4 (p. 5) and table 2 (``sur_store``). **Departs from the
    manual:** eq. 4 reads ``Xb·Zs·POR·(Wb/4 + Zs/2α)``, which is inconsistent
    with the manual's own geometry; see the reconciliation notes, §2.1.
    Callers must keep ``zs <= ws·alpha``; this function does not clamp.
    """
    return lb * (2 * ws * zs - zs**2 / alpha) * por


@njit(cache=True)
def surface_thickness_from_storage(lb, alpha, ws, por, ss):
    """Saturated thickness Zs for surface storage Ss [m].

    Root of ``Zs²/(2α) − 2·ws·Zs + Ss/(Lb·POR) = 0``; returns ``ws·α`` when
    the discriminant is negative.

    Manual: table 2 (``sur_z``), described as inverting eq. 4.
    **Not the inverse of `surface_storage_from_thickness`** (suspected issue
    M10, inherited from the USGS R code): the quadratic coefficient is 1/(2α)
    where the storage formula implies 1/α. The round trip returns 0.586·Zs at
    full saturation. Kept as-is pending a decision.
    """
    a1 = 1 / (2 * alpha)
    b1 = -2 * ws
    c1 = ss / (lb * por)
    discriminant = b1**2 - 4 * a1 * c1
    if discriminant < 0:
        return ws * alpha
    return (-b1 - math.sqrt(discriminant)) / (2 * a1)


@njit(cache=True)
def surface_discharge(lb, alpha, ks, zs):
    """Discharge from the surface reservoir, Qs [m³/day].

    ``Qs = 2·Lb·Ks·α·Zs``: Darcy flow with gradient α through the sections
    at ±Xs, each Lb wide and Zs thick.

    Manual: eq. 9 (p. 8), table 1. Matches exactly. (Table 2 cites "equation
    11" for ``sur_q``; that is a cross-reference error.)
    """
    return 2 * lb * zs * alpha * ks


@njit(cache=True)
def saturated_area_runoff(lb, alpha, zs, impulse):
    """Direct runoff of an impulse falling on saturated land, Qd [m³/day].

    ``Qd = 2·Lb·(Zs/α)·I``: the impulse depth times the saturated strip
    (Xs = Zs/α wide on each side of the channel).

    Manual: eq. 19 (p. 12). Matches exactly. Eq. 15 writes the same flux with
    Wb in place of Lb, which conflicts with eq. 19 and the geometry.
    """
    return 2 * lb * zs / alpha * impulse


@njit(cache=True)
def infiltration_rate(lb, ws, ks, alpha, zs, impulse):
    """Infiltration of an impulse through unsaturated land, F [m³/day].

    ``F = 2·Lb·(ws − Zs/α)·min(I, Ks)``: the unsaturated land area times the
    impulse, capped at Ks. Callers limit F to the available surface storage.

    Manual: eq. 14 (p. 8). **Departs from the manual**, which has Xb in place
    of Lb and caps the rate at Kz rather than Ks. The manual contradicts
    itself here: tables 2 and 5 agree with the code (Lb, Ks).
    """
    return 2 * lb * (ws - zs / alpha) * min(impulse, ks)


@njit(cache=True)
def recharge_rate(lb, xb, ws, kz, zs, por):
    """Recharge from the surface to the base reservoir, R [m³/day].

    ``R = (Lb − Xb)·Wb·min(POR·Zs, Kz)``: the base reservoir's unsaturated
    area upstream of Xb, times the drainable water depth capped at Kz.
    Callers limit R to the available base storage plus baseflow.

    Manual: eq. 16 (p. 9), table 1. Matches exactly. The manual compares a
    depth with a rate (Kz) here too, which assumes a time step of one unit.
    """
    return (lb - xb) * 2 * ws * min(zs * por, kz)
