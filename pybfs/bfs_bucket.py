# -*- coding: utf-8 -*-
"""Simplified two-bucket baseflow separation (experimental).

A geometry-free reformulation of the PyBFS reservoir model. Both reservoirs
become lumped volumetric buckets (storage in m3, flux in m3/day), so none of
the geometry inputs (lb, wb, x1, por, alpha, beta) are needed:

    Surface bucket (state Ss):
        Qs    = ks * Ss                 discharge to stream        (ks, 1/day)
        Qr    = kr * Ss                 recharge/leakage to base   (kr, 1/day)
        infil = min(W, Ss_max - Ss)     W = inferred precip volume (m3/day)
        Qd    = W - infil               direct runoff = saturation excess only

    Base bucket (state Sb):
        Qb = k * Sb**n                  nonlinear reservoir        (k, n)

    Qsim = Qs + Qb + Qd

During recessions W = 0, so the base bucket decays as dS/dt = -k*S**n
(exponential when n == 1). During rises W is inferred by a bounded solve so the
trapezoidal mid-point Qsim matches observed flow -- replacing the original
nested impulse-estimation loop.

This module is intentionally standalone and does not touch bfs.py / utilities.py.
Output columns mirror bfs() so plot.py / skill.py can consume the result;
geometry-only columns (X, Zs.L, Zb.L) are filled with NaN.
"""

import numpy as np
import pandas as pd


def _detect_rise_recess(qin, prec, fr4rise, timestep='day'):
    """Rise / recession flags and consecutive-recession counter.

    Mirrors the logic in bfs.bfs() (which is geometry-independent) so the two
    models see the same rising limbs and recession weighting.
    """
    p = len(qin)
    dq = np.zeros(p)
    if timestep == 'day':
        dq[1:] = qin[1:] - qin[:-1]
    elif timestep == 'hour':
        for y in range(24, p):
            dq[y] = qin[y] - np.nanmax(qin[(y - 24):y])

    dqfr = np.zeros_like(dq, dtype=float)
    mask_zero_both = (dq == 0) & (qin == 0)
    mask_neg_dq_zero_qin = (dq < 0) & (qin == 0)
    mask_valid = qin != 0
    dqfr[mask_zero_both] = 0
    dqfr[mask_neg_dq_zero_qin] = 1
    dqfr[mask_valid] = dq[mask_valid] / qin[mask_valid]

    rise = (dqfr > fr4rise) & (dq > prec)
    rise[np.isnan(rise)] = False
    recess = (dqfr <= fr4rise) | (dq < prec)
    recess[np.isnan(recess)] = False

    recess_day = np.cumsum(recess) - np.maximum.accumulate(
        (~recess).astype(int) * np.cumsum(recess)
    )
    return rise, recess_day


def _qsim_given_w(W, Ss_in, qr, qs_in, qb_mid, ss_max, ks):
    """Trapezoidal Qsim (surface + direct runoff, base held fixed) for a given
    precip volume W. Monotonically increasing in W, so bisection-solvable.

    Returns (qsim, infil, qd, qs_mid, ss_en_prov).
    """
    ssa = max(ss_max - Ss_in, 0.0)
    infil = min(W, ssa)
    qd = W - infil
    ss_en_prov = max(Ss_in + infil - qs_in - qr, 0.0)
    qs_en = ks * ss_en_prov
    qs_mid = (qs_in + qs_en) / 2.0
    qsim = qs_mid + qb_mid + qd
    return qsim, infil, qd, qs_mid, ss_en_prov


def bfs_bucket(streamflow, params, flow, basin_area=None,
               timestep='day', error_basis='total'):
    """Run the simplified two-bucket baseflow separation.

    Parameters
    ----------
    streamflow : pd.DataFrame
        Columns 'Date' (datetime) and 'Streamflow' (m3/day).
    params : dict
        {'Ss_max', 'ks', 'kr', 'k', 'n'}:
        - Ss_max : surface storage capacity (m3)
        - ks     : surface discharge coefficient (1/day)
        - kr     : recharge coefficient surface->base (1/day)
        - k      : base discharge coefficient (units depend on n)
        - n      : base nonlinearity exponent (1 => exponential recession)
    flow : list
        Flow metrics [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise]; only Rs (index 1),
        Prec (index 4) and Frac4Rise (index 5) are used.
    basin_area : float, optional
        Basin area (m2). Used only to report Impulse.L as a depth; dynamics are
        volumetric. If None, Impulse.L is reported as a volume (m3/day).
    timestep : str, optional
        'day' or 'hour' (default 'day').
    error_basis : str, optional
        'base' or 'total' (default 'total'), matching bfs().

    Returns
    -------
    pd.DataFrame
        Same column schema as bfs(); X, Zs.L, Zb.L are NaN (no geometry).
    """
    Ss_max = float(params['Ss_max'])
    ks = float(params['ks'])
    kr = float(params['kr'])
    k = float(params['k'])
    n = float(params['n'])

    rs = flow[1]
    prec = flow[4]
    fr4rise = flow[5]

    date = pd.to_datetime(streamflow['Date'])
    qin = np.array(streamflow['Streamflow'], dtype=float)
    qin[qin < 0] = np.nan
    qmean = np.nanmean(qin)
    p = len(qin)

    rise, recess_day = _detect_rise_recess(qin, prec, fr4rise, timestep)

    # Outputs
    qcomp = np.full((p, 3), np.nan)   # surface, base, direct runoff
    ST = np.full((p, 2), np.nan)      # Ss, Sb
    EXC = np.full((p, 2), np.nan)     # infil (into surface), recharge (to base)
    I = np.full(p, np.nan)            # inferred precip
    ETA = np.full(p, np.nan)

    # First valid time step
    ts = 0
    while ts < p and np.isnan(qin[ts]):
        ts += 1
    if ts >= p:
        return _empty_frame(date, qin, p)

    # Initialization from observed flow at first valid step
    qb_in = min(qin[ts], qmean)
    Sb_in = (qb_in / k) ** (1.0 / n) if qb_in > 0 else 0.0
    qs_in0 = max(qin[ts] - qb_in, 0.0)
    Ss_in = min(qs_in0 / ks, Ss_max) if ks > 0 else 0.0

    ts_ini = True
    while ts < p:
        if not ts_ini:
            Ss_in = ST[ts - 1, 0]
            Sb_in = ST[ts - 1, 1]

        qs_in = ks * Ss_in
        qr_in = kr * Ss_in
        qb_in = k * Sb_in ** n

        qob = qin[ts]

        # --- base bucket (independent of W within the step) ---
        # recharge uses start-of-step surface storage; keeps the W-solve monotonic
        qr_mid = qr_in
        Sb_en = max(Sb_in + qr_mid - qb_in, 0.0)
        qb_en = k * Sb_en ** n
        qb_mid = (qb_in + qb_en) / 2.0

        # --- infer precip volume W on rising limbs ---
        W = 0.0
        infil = 0.0
        qd = 0.0
        if not ts_ini and not np.isnan(qob):
            etaest = qob - qb_mid - qs_in
            is_rise = rise[ts] or (ts > 0 and rise[ts - 1])
            if is_rise and etaest > max(prec, qob / 100.0):
                # Qsim(W) is monotone increasing; bisect for Qsim == qob.
                W = _solve_w(qob, Ss_in, qr_mid, qs_in, qb_mid, Ss_max, ks, prec)
                _, infil, qd, _, _ = _qsim_given_w(
                    W, Ss_in, qr_mid, qs_in, qb_mid, Ss_max, ks
                )

        # --- surface bucket end-of-step (consistent with chosen W) ---
        ss_en_prov = max(Ss_in + infil - qs_in - qr_in, 0.0)
        qs_en = ks * ss_en_prov
        qs_mid = (qs_in + qs_en) / 2.0

        Ss_new = Ss_in + infil - qs_mid - qr_mid
        Ss_new = min(max(Ss_new, 0.0), Ss_max)
        Sb_new = max(Sb_in + qr_mid - qb_mid, 0.0)

        # --- record ---
        qcomp[ts, 0] = qs_mid
        qcomp[ts, 1] = qb_mid
        qcomp[ts, 2] = 0.0 if ts_ini else qd
        EXC[ts, 0] = infil
        EXC[ts, 1] = qr_mid
        ST[ts, 0] = Ss_new
        ST[ts, 1] = Sb_new
        I[ts] = W
        ETA[ts] = qob - np.nansum(qcomp[ts, 0:3])

        ts += 1
        ts_ini = False

    # Impulse as depth if area supplied
    if basin_area:
        I = I / basin_area

    qsim = np.sum(qcomp, axis=1)
    qsim[np.isnan(qcomp).any(axis=1)] = np.nan

    if error_basis == 'base':
        q4er = qcomp[:, 1]
    else:
        q4er = qsim
    APE = (q4er - qin) / (qin + prec)
    APE[(qin == 0) & (q4er == 0)] = 0
    Weight = 1 - np.exp(rs * recess_day)
    Weight[APE > 0] = 1

    tmp = pd.DataFrame({
        'Date': date, 'Qob': qin, 'Qsim': qsim,
        'SurfaceFlow': qcomp[:, 0], 'Baseflow': qcomp[:, 1],
        'DirectRunoff': qcomp[:, 2], 'X': np.nan, 'Eta': ETA,
        'StSur': ST[:, 0], 'StBase': ST[:, 1], 'Impulse.L': I,
        'Zs.L': np.nan, 'Zb.L': np.nan, 'Infil': EXC[:, 0], 'Rech': EXC[:, 1],
        'RecessCount.T': recess_day, 'AdjPctEr': APE, 'Weight': Weight,
    })
    return tmp[['Date', 'Qob', 'Qsim', 'SurfaceFlow', 'Baseflow', 'DirectRunoff',
                'X', 'Eta', 'StSur', 'StBase', 'Impulse.L', 'Zs.L', 'Zb.L',
                'Infil', 'Rech', 'RecessCount.T', 'AdjPctEr', 'Weight']]


def _solve_w(qob, Ss_in, qr_mid, qs_in, qb_mid, ss_max, ks, prec, iters=60):
    """Bisection for the precip volume W such that Qsim(W) == qob.

    Qsim is monotonically increasing in W, so a simple bracket-and-bisect is
    robust. Upper bracket grows until Qsim(hi) >= qob.
    """
    lo, hi = 0.0, max(qob, prec)
    for _ in range(40):
        qsim_hi, *_ = _qsim_given_w(hi, Ss_in, qr_mid, qs_in, qb_mid, ss_max, ks)
        if qsim_hi >= qob:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        qsim_mid, *_ = _qsim_given_w(mid, Ss_in, qr_mid, qs_in, qb_mid, ss_max, ks)
        if qsim_mid < qob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _empty_frame(date, qin, p):
    """All-NaN result frame (no valid streamflow)."""
    nan = np.full(p, np.nan)
    return pd.DataFrame({
        'Date': date, 'Qob': qin, 'Qsim': nan, 'SurfaceFlow': nan,
        'Baseflow': nan, 'DirectRunoff': nan, 'X': nan, 'Eta': nan,
        'StSur': nan, 'StBase': nan, 'Impulse.L': nan, 'Zs.L': nan,
        'Zb.L': nan, 'Infil': nan, 'Rech': nan, 'RecessCount.T': nan,
        'AdjPctEr': nan, 'Weight': nan,
    })
