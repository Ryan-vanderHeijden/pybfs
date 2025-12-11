# -*- coding: utf-8 -*-
"""Calibration functions for PyBFS

Functions for calibrating the baseflow separation model parameters using
optimization techniques to match observed streamflow behavior.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .bfs import bfs
from .utilities import base_table, bf_ci, flow_metrics

# Global variable to store bfs_out (matching R behavior)
bfs_out = None


def calculate_error(bfs_out_df):
    """Calculate Error as mean absolute weighted APE (matching R bfs() return value)
    
    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs() function
    
    Returns
    -------
    float
        Mean absolute weighted APE (matching R bfs() return value)
    """
    ape_col = 'AdjPctEr' if 'AdjPctEr' in bfs_out_df.columns else 'AdjPctEr'
    weight_col = 'Weight' if 'Weight' in bfs_out_df.columns else 'Weight'
    APE = bfs_out_df[ape_col].values
    Weight = bfs_out_df[weight_col].values
    # R bfs() returns: sum(abs(APE*Weight))/sum(Weight)
    if np.nansum(Weight) > 0:
        return np.nansum(np.abs(APE * Weight)) / np.nansum(Weight)
    else:
        return np.nan


def objective(bfs_out_df, prec):
    """Objective function for calibration

    Calculates the objective function value for calibration optimization.
    Penalizes over-prediction and weights errors by recession length.

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs() function
    prec : float
        Precision threshold

    Returns
    -------
    float
        Objective function value (lower is better)
    """
    # Handle column name variations
    ape_col = 'AdjPctEr' if 'AdjPctEr' in bfs_out_df.columns else 'AdjPctEr'
    weight_col = 'Weight' if 'Weight' in bfs_out_df.columns else 'Weight'
    qob_col = 'Qob' if 'Qob' in bfs_out_df.columns else 'Qob.L3'
    baseflow_col = 'Baseflow' if 'Baseflow' in bfs_out_df.columns else 'Baseflow.L3'

    APE = bfs_out_df[ape_col].values
    Weight = bfs_out_df[weight_col].values.copy()

    # Days when error is positive (over prediction) do not count
    Weight[APE > prec / (bfs_out_df[qob_col].values + prec)] = 0

    # Limit influence of hydroperiod for non-perennial streams
    Weight[(bfs_out_df[qob_col].values == 0) & (bfs_out_df[baseflow_col].values == 0)] = 0

    OBJ = np.nansum(Weight * (-1 + APE**2))
    return OBJ


def ini_params(area, lb, x1, wb, por, beta, rb1, tmp_q):
    """Set initial values of Lb, Wb, and Kb

    Initializes basin length, width, and base hydraulic conductivity to match
    recession rate and allow baseflow to equal mean flow while Kb has physically
    possible value.

    Parameters
    ----------
    area : float
        Drainage area (m²)
    lb : float
        Initial basin length (m)
    x1 : float
        Scaling parameter for base length (m)
    wb : float
        Initial base width (m)
    por : float
        Porosity (dimensionless)
    beta : float
        Base surface exponent
    rb1 : float
        Baseflow recession coefficient at mean flow [1/day]
    tmp_q : array-like
        Streamflow time series (m³/day)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Lb, Wb, Kb containing optimized initial parameters
    """
    qmean = np.nanmean(tmp_q)

    # Iterate to ensure Kb is within physical bounds
    iterate = True
    while iterate:
        iterate = False
        xb = np.linspace(0, lb, 1001)
        z = (xb / x1) ** beta
        dzdx = beta / (x1 ** beta) * xb ** (beta - 1)

        # Check Kb bounds
        kb_calc = -rb1 * (por * (lb - xb / 2) * x1 ** beta) / (beta * xb ** (beta - 1))
        kb_calc[xb == 0] = np.nan  # Avoid division by zero

        if np.any(kb_calc[~np.isnan(kb_calc)] < 1e-7):
            lb = lb * 1.1
            iterate = True
        if np.any(kb_calc[~np.isnan(kb_calc)] > 1e4):
            lb = lb * 0.9
            iterate = True

    # Adjust wb so that Xb is between 0.1 and 0.9 * Lb when baseflow equals mean streamflow
    iterate = True
    while iterate:
        iterate = False
        xb = np.linspace(0, lb, 1001)
        z = (xb / x1) ** beta
        dzdx = beta / (x1 ** beta) * xb ** (beta - 1)
        sb = wb * por * (lb - xb / 2) * (xb / x1) ** beta
        kb = -rb1 * (por * (lb - xb / 2) * x1 ** beta) / (beta * xb ** (beta - 1))
        kb[xb == 0] = np.nan

        # Baseflow vector corresponding to Z vector
        qb = wb * z * kb * dzdx
        qb[np.isnan(kb)] = np.nan

        # Find index where qb > qmean
        qmean_idx = np.where(qb > qmean)[0]
        if len(qmean_idx) == 0:
            qmean_idx = len(qb) - 1
        else:
            qmean_idx = qmean_idx[0]

        if xb[qmean_idx] < (0.1 * lb):
            wb = 0.9 * wb  # Reduce Wb so Xb is further from outlet
            if wb > 10:
                iterate = True
        if xb[qmean_idx] > (0.9 * lb):
            wb = 1.1 * wb  # Increase Wb so Xb is closer to outlet
            if wb * area > lb:
                lb = 0.9 * area / wb
                iterate = True

    kb = qb[qmean_idx] / (wb * z[qmean_idx] * dzdx[qmean_idx])

    return pd.DataFrame({'Lb': [lb], 'Wb': [wb], 'Kb': [kb]})


def cal_initial(logx, tmp_q, dys, timestep, error_basis, basin_char, gw_hyd, flow, qmean):
    """Initial calibration of Lb, Wb, ALPHA, Ks, Kb, Kz

    Objective function for initial calibration step using log-transformed parameters.

    Parameters
    ----------
    logx : array-like
        Log10 of parameters [Lb, Wb, ALPHA, Ks, Kb, Kz]
    tmp_q : array-like
        Streamflow time series (m³/day)
    dys : array-like
        Date vector
    timestep : str
        'day' or 'hour'
    error_basis : str
        'base' or 'total'
    basin_char : list
        Basin characteristics [area, lb, x1, wb, por]
    gw_hyd : list
        Groundwater hydraulic parameters [alpha, beta, ks, kb, kz]
    flow : list
        Flow metrics [qthresh, rs, rb1, rb2, prec, fr4rise]
    qmean : float
        Mean streamflow (m³/day)

    Returns
    -------
    float
        Objective function value (100 for bad parameters, otherwise objective)
    """
    global bfs_out

    # Check for bad parameter values
    lb = 10 ** logx[0]
    wb = 10 ** logx[1]
    alpha = 10 ** logx[2]
    ks = 10 ** logx[3]
    kb = 10 ** logx[4]
    kz = 10 ** logx[5]

    bad_X = [
        (lb * wb) > basin_char[0],  # Width x Length > Area
        logx[2] > -1,  # Surface slope > 10%
        logx[3] < -8,  # Ks < 10^-8
        logx[3] > 5,   # Ks > 10^5
        logx[4] < -8,  # Kb < 10^-8
        logx[4] > 5,   # Kb > 10^5
        logx[5] < -8,  # Kz < 10^-8
        logx[5] > 5    # Kz > 10^5
    ]

    if any(bad_X):
        return 100.0

    # Update parameters
    basin_char_new = basin_char.copy()
    basin_char_new[1] = lb  # Lb
    basin_char_new[3] = wb  # Wb

    gw_hyd_new = gw_hyd.copy()
    gw_hyd_new[0] = alpha  # ALPHA
    gw_hyd_new[2] = ks     # Ks
    gw_hyd_new[3] = kb     # Kb
    gw_hyd_new[4] = kz     # Kz

    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })

    # Generate baseflow table
    SBT = base_table(basin_char_new[1], basin_char_new[2], basin_char_new[3],
                     gw_hyd_new[1], gw_hyd_new[3], streamflow_df, basin_char_new[4])

    # Run BFS
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow)
    obj = objective(bfs_out, prec=flow[4])

    return obj


def cal_basetable(x, b, params, tmp_q):
    """Calibrate Lb, X1, Wb, and Kb for non-linear surface of base reservoir

    Objective function for calibrating base reservoir parameters to match
    recession rates across flow quantiles.

    Parameters
    ----------
    x : array-like
        Parameters [Lb, X1, Wb, Kb]
    b : float
        Beta parameter
    params : list
        [AREA, POR, Qmean, Qthresh, RbI, RbS]
    tmp_q : array-like
        Streamflow time series (m³/day)

    Returns
    -------
    float
        Objective function value (1000 for bad parameters, otherwise mean absolute error)
    """
    area = params[0]
    por = params[1]
    qmean = params[2]
    qthresh = params[3]
    rbi = params[4]
    rbs = params[5]

    # Check for bad parameters
    # X1 should have a reasonable minimum (at least 0.1) to prevent unrealistic values
    bad_X = [
        np.any(np.isinf(x)),
        x[0] <= 0,
        x[0] > area / x[2],
        x[1] <= 0.1,  # Minimum X1 to prevent unrealistic small values
        x[2] > area / x[0],
        x[3] < 1e-8,
        x[3] > 1e5
    ]

    if any(bad_X):
        return 1000.0

    lb = x[0]
    x1 = x[1]
    wb = x[2]
    kb = x[3]

    # Create streamflow DataFrame for base_table
    streamflow_df = pd.DataFrame({'Streamflow': tmp_q})
    sbt = base_table(lb, x1, wb, b, kb, streamflow_df, por)

    # Check for viable solution
    if np.nanmax(sbt['Q']) < qthresh:
        return 1000.0

    # Recession coefficient for each interval
    r = sbt['Q'] / sbt['S']
    r[np.isnan(r)] = 0
    r[np.isinf(r)] = 0

    # Index for deciles of range between Qmin and Qmean
    qmin = np.nanmin(tmp_q[tmp_q > 0]) / 10
    tmp_qq = 10 ** np.linspace(np.log10(qmin), np.log10(qmean), 101)

    # Convert to numpy arrays for faster operations
    sbt_q = sbt['Q'].values
    sbt_s = sbt['S'].values
    
    # Find matching indices in SBT using vectorized searchsorted
    # For each qq, find the last index where sbt_q <= qq
    tmp_indices = []
    for qq in tmp_qq:
        # Use searchsorted to find insertion point (faster than comparison)
        idx = np.searchsorted(sbt_q, qq, side='right')
        if idx > 0:
            tmp_indices.append(idx - 1)

    # Filter indices using vectorized operations
    if len(tmp_indices) > 0:
        tmp_indices = np.array(tmp_indices)
        # First filter: Q > 0 (must be positive for log10)
        sbt_q_values = sbt_q[tmp_indices]
        valid_q = sbt_q_values > 0
        # Only compute log10 on positive values
        if np.any(valid_q):
            valid_recess = np.zeros(len(tmp_indices), dtype=bool)
            valid_recess[valid_q] = (rbi + rbs * np.log10(sbt_q_values[valid_q])) < 0
            tmp_indices = tmp_indices[valid_q & valid_recess]
            tmp_indices = np.unique(tmp_indices)
        else:
            tmp_indices = []
    else:
        tmp_indices = []

    # Objective is matching recession rates for deciles from Qthresh to Qmean
    if len(tmp_indices) < 10:
        return 100.0

    # Use vectorized operations
    r_subset = r[tmp_indices]
    q_subset = sbt_q[tmp_indices]
    obj = np.nanmean(np.abs(r_subset + (rbi + rbs * np.log10(q_subset))))
    if np.isinf(obj) or np.isnan(obj):
        obj = 100.0

    return obj


def cal_base(x, tmp_q, dys, timestep, error_basis, basin_char, gw_hyd, flow):
    """Calibrate basin length, base width, base conductivity, and recharge

    Objective function for calibrating base reservoir parameters.

    Parameters
    ----------
    x : array-like
        Parameters [Lb, Wb, Kb, Kz]
    tmp_q : array-like
        Streamflow time series (m³/day)
    dys : array-like
        Date vector
    timestep : str
        'day' or 'hour'
    error_basis : str
        'base' or 'total'
    basin_char : list
        Basin characteristics [area, lb, x1, wb, por]
    gw_hyd : list
        Groundwater hydraulic parameters [alpha, beta, ks, kb, kz]
    flow : list
        Flow metrics [qthresh, rs, rb1, rb2, prec, fr4rise]

    Returns
    -------
    float
        Objective function value (100 for bad parameters, otherwise objective)
    """
    global bfs_out

    # Check for bad parameters
    bad_X = [
        x[0] * x[1] > basin_char[0],
        x[0] < 0,
        x[1] < 0,
        x[2] < 1e-8,
        x[2] > 1e5,
        x[3] < 1e-8,
        x[3] > 1e5
    ]

    if any(bad_X):
        return 100.0

    lb = x[0]
    wb = x[1]
    kb = x[2]
    kz = x[3]

    basin_char_new = basin_char.copy()
    basin_char_new[1] = lb
    basin_char_new[3] = wb

    gw_hyd_new = gw_hyd.copy()
    gw_hyd_new[3] = kb
    gw_hyd_new[4] = kz

    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })

    # Generate baseflow table
    SBT = base_table(basin_char_new[1], basin_char_new[2], basin_char_new[3],
                     gw_hyd_new[1], gw_hyd_new[3], streamflow_df, basin_char_new[4])

    # Run BFS
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow)
    obj = objective(bfs_out, prec=flow[4])

    return obj


def cal_surface(x, tmp_q, dys, timestep, error_basis, basin_char, gw_hyd, flow):
    """Calibrate surface parameters

    Objective function for calibrating surface reservoir parameters.

    Parameters
    ----------
    x : array-like
        Log10 of parameters [Wb, ALPHA, Ks, Kz]
    tmp_q : array-like
        Streamflow time series (m³/day)
    dys : array-like
        Date vector
    timestep : str
        'day' or 'hour'
    error_basis : str
        'base' or 'total'
    basin_char : list
        Basin characteristics [area, lb, x1, wb, por]
    gw_hyd : list
        Groundwater hydraulic parameters [alpha, beta, ks, kb, kz]
    flow : list
        Flow metrics [qthresh, rs, rb1, rb2, prec, fr4rise]

    Returns
    -------
    float
        Objective function value (100 for bad parameters, otherwise objective)
    """
    global bfs_out

    # Check for bad parameters
    bad_X = [
        10 ** x[0] > basin_char[0] / basin_char[1],
        x[1] > -1,
        x[2] < -8,
        x[2] > 5,
        x[3] < -8,
        x[3] > 5
    ]

    if any(bad_X):
        return 100.0

    wb = 10 ** x[0]
    alpha = 10 ** x[1]
    ks = 10 ** x[2]
    kz = 10 ** x[3]

    basin_char_new = basin_char.copy()
    basin_char_new[3] = wb

    gw_hyd_new = gw_hyd.copy()
    gw_hyd_new[0] = alpha
    gw_hyd_new[2] = ks
    gw_hyd_new[4] = kz

    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })

    # Generate baseflow table
    SBT = base_table(basin_char_new[1], basin_char_new[2], basin_char_new[3],
                     gw_hyd_new[1], gw_hyd_new[3], streamflow_df, basin_char_new[4])

    # Run BFS
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow)
    obj = objective(bfs_out, prec=flow[4])

    return obj


def bfs_calibrate(tmp_site, tmp_area, tmp_q, dys):
    """Calibrate a site

    Main calibration function that performs multi-step optimization to find
    optimal parameters for baseflow separation.

    Parameters
    ----------
    tmp_site : str
        Site identification string
    tmp_area : float
        Drainage area (m²)
    tmp_q : array-like
        Streamflow time series (m³/day)
    dys : array-like
        Date vector (can be strings or datetime objects)

    Returns
    -------
    tuple
        (bf_params, bff, ci_table, bfs_out) where:
        - bf_params: DataFrame with calibrated parameters
        - bff: DataFrame with baseflow fractions
        - ci_table: DataFrame with credible intervals
        - bfs_out: DataFrame with full BFS output
    """
    global bfs_out

    # Run flow_metrics function
    try:
        flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    except Exception as e:
        print(f"Error in flow_metrics: {e}")
        import traceback
        traceback.print_exc()
        return (None, None, None, None)

    # Check if flow_metrics returned valid values
    if flow_result is None or len(flow_result) < 6:
        print("flow_metrics returned None or insufficient values")
        return (None, None, None, None)
    
    # Check each value individually for debugging
    invalid_indices = []
    for i, val in enumerate(flow_result[:6]):
        if np.isnan(val) or np.isinf(val):
            invalid_indices.append(i)
    
    # Debug output (commented out for production)
    # param_names = ['Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Fr4Rise']
    # print(f"    flow_metrics results:")
    # for i, name in enumerate(param_names):
    #     print(f"      {name}: {flow_result[i]}")
    
    if invalid_indices:
        param_names = ['Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Fr4Rise']
        print(f"    flow_metrics returned invalid values at indices {invalid_indices}:")
        for idx in invalid_indices:
            print(f"      {param_names[idx]}: {flow_result[idx]}")
        return (None, None, None, None)

    Qthresh = flow_result[0]
    Rs = flow_result[1]
    Rb1 = flow_result[2]
    Rb2 = flow_result[3]
    Prec = flow_result[4]
    Frac4Rise = flow_result[5]
    flow = [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise]

    Qmean = np.nanmean(tmp_q[tmp_q >= 0])

    # Get rb10 from flow_metrics (7th element)
    if len(flow_result) > 6 and not np.any(np.isnan(flow_result[6])):
        rb10 = flow_result[6]
        RbI = rb10[0]  # Intercept
        RbS = rb10[1]  # Slope
    else:
        # Default values if rb10 not available
        RbI = -0.01
        RbS = -0.001

    # Initialize parameters with nominal estimates
    Lb = 2 * (tmp_area / 2) ** 0.5  # Basin length
    Wb = tmp_area / Lb / 10  # Width of base
    Ws = Wb / 2
    POR = 0.15  # Drainable porosity

    ALPHA = 0.01  # Surface hydraulic gradient
    BETA = 1  # Base surface exponent
    X1 = 1 / ALPHA  # Base gradient equals surface gradient

    tmp = ini_params(tmp_area, Lb, X1, Wb, POR, BETA, Rb1, tmp_q)
    Lb = tmp['Lb'].iloc[0]
    Wb = tmp['Wb'].iloc[0]
    Kb = tmp['Kb'].iloc[0]

    Ks = (1 - np.exp(Rs)) * POR * (3/4 * Ws) / ALPHA  # Surface hydraulic conductivity
    Kz = 10 * Qmean / (Lb * Wb)  # Vertical hydraulic conductivity

    basin_char = [tmp_area, Lb, X1, Wb, POR]
    gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })

    # Generate baseflow table
    SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                     gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])

    # Initial BFS run
    bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)
    Error = calculate_error(bfs_out)

    # Calculate BFF
    baseflow_col = 'Baseflow' if 'Baseflow' in bfs_out.columns else 'Baseflow.L3'
    qob_col = 'Qob' if 'Qob' in bfs_out.columns else 'Qob.L3'
    BFF = np.nansum(bfs_out[baseflow_col]) / np.nansum(bfs_out[qob_col])

    bf_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Lb': [Lb],
        'X1': [X1],
        'Wb': [Wb],
        'POR': [POR],
        'ALPHA': [ALPHA],
        'BETA': [BETA],
        'Ks': [Ks],
        'Kb': [Kb],
        'Kz': [Kz],
        'Qthresh': [Qthresh],
        'Rs': [Rs],
        'Rb1': [Rb1],
        'Rb2': [Rb2],
        'Prec': [Prec],
        'Frac4Rise': [Frac4Rise],
        'Error': [Error],
        'BFF': [BFF]
    })

    # STEP 1: CALIBRATE ASSUMING BETA = 1
    print("  Step 1: Initial calibration (beta=1)...")
    X = np.array([Lb, Wb, ALPHA, Ks, Kb, Kz])
    LOGX = np.log10(X)

    result = minimize(
        cal_initial,
        LOGX,
        args=(tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean),
        method='Nelder-Mead',
        options={'maxiter': 500}  # Reduced for speed
    )

    if np.all(np.isfinite(result.x)):
        Lb = 10 ** result.x[0]
        Wb = 10 ** result.x[1]
        ALPHA = 10 ** result.x[2]
        Ks = 10 ** result.x[3]
        Kb = 10 ** result.x[4]
        Kz = 10 ** result.x[5]
        
        # Update X1 to match new ALPHA value (X1 = 1/ALPHA)
        X1 = 1.0 / ALPHA

        basin_char = [tmp_area, Lb, X1, Wb, POR]
        gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

        X = np.array([Lb, Wb, ALPHA, Ks])
        LOGX = np.log10(X)

        result = minimize(
            cal_surface,
            LOGX,
            args=(tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow),
            method='Nelder-Mead',
            options={'maxiter': 500}  # Reduced for speed
        )

        if np.all(np.isfinite(result.x)):
            Lb = 10 ** result.x[0]
            Wb = 10 ** result.x[1]
            ALPHA = 10 ** result.x[2]
            Ks = 10 ** result.x[3]
            
            # Update X1 to match new ALPHA value (X1 = 1/ALPHA)
            X1 = 1.0 / ALPHA

            basin_char = [tmp_area, Lb, X1, Wb, POR]
            gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

            SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                           gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
            bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)
            Error = calculate_error(bfs_out)

            tmp_bf = bfs_out[baseflow_col].copy()
            tmp_ov = bfs_out[baseflow_col] > bfs_out[qob_col]
            tmp_ov[np.isnan(tmp_ov)] = False
            tmp_bf[tmp_ov] = bfs_out[qob_col][tmp_ov]
            BFF = np.nansum(tmp_bf[~np.isnan(bfs_out[qob_col])]) / np.nansum(bfs_out[qob_col][~np.isnan(bfs_out[qob_col])])

            bf_params = pd.concat([bf_params, pd.DataFrame({
                'tmp.site': [tmp_site],
                'tmp.area': [tmp_area],
                'Lb': [Lb],
                'X1': [X1],
                'Wb': [Wb],
                'POR': [POR],
                'ALPHA': [ALPHA],
                'BETA': [BETA],
                'Ks': [Ks],
                'Kb': [Kb],
                'Kz': [Kz],
                'Qthresh': [Qthresh],
                'Rs': [Rs],
                'Rb1': [Rb1],
                'Rb2': [Rb2],
                'Prec': [Prec],
                'Frac4Rise': [Frac4Rise],
                'Error': [Error],
                'BFF': [BFF]
            })], ignore_index=True)

    # STEP 2: CALIBRATE NON-LINEAR BASEFLOW FUNCTION FOR RECESSION RATES AT BETA = 1 TO 20
    print("  Step 2: Calibrating non-linear baseflow function (testing beta values)...")
    tmp_out = []

    n = 0
    b = 0.5
    continue_cal = True
    last_obj = None

    while continue_cal:
        b = b + 0.1

        # X1 so that Qb=QMEAN @ Xb=Lb/2
        if (2 * b - 1) != 0:
            X1 = (Wb * Kb * b ** 2 / ((2 * b - 1) * Qmean) * (Lb / 2) ** (2 * b - 1)) ** (1 / (2 * b))
        else:
            X1 = np.nan

        # Ensure X1 has a reasonable minimum value (at least 1.0)
        # If calculated X1 is too small or invalid, use relationship X1 = 1/ALPHA
        if not np.isfinite(X1) or X1 < 1.0:
            X1 = max(1.0 / ALPHA, 1.0)

        if np.isfinite(X1) and X1 > 0:
            X = np.array([Lb, X1, Wb, Kb])

            result = minimize(
                cal_basetable,
                X,
                args=(b, [tmp_area, POR, Qmean, Qthresh, RbI, RbS], tmp_q),
                method='Nelder-Mead',
                options={'maxiter': 500}  # Reduced from 1000 for speed
            )

            if np.all(np.isfinite(result.x)):
                n = n + 1
                tmp_out.append([result.x[0], result.x[1], result.x[2], b, result.x[3], result.fun])
                last_obj = result.fun
                if n % 5 == 0:
                    print(f"    Tested beta={b:.1f}, objective={result.fun:.4f}")

        if b > 10:
            if len(tmp_out) >= 2 and last_obj is not None:
                if abs((last_obj - tmp_out[n-1][5]) / last_obj) < 0.001:
                    print(f"    Convergence reached at beta={b:.1f}")
                    continue_cal = False
        if b >= 20:
            continue_cal = False

    if len(tmp_out) > 0:
        tmp_out = np.array(tmp_out)
        best_idx = np.argmin(tmp_out[:, 5])
        X = tmp_out[best_idx, :5]

        Lb = X[0]
        X1 = X[1]
        Wb = X[2]
        basin_char = [tmp_area, Lb, X1, Wb, POR]

        BETA = X[3]
        Kb = X[4]
        
        print(f"    Selected best beta={BETA:.3f} with objective={tmp_out[best_idx, 5]:.4f}")
        gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

        X = np.array([Lb, Wb, Kb, Kz])

        print("    Optimizing base parameters...")
        result = minimize(
            cal_base,
            X,
            args=(tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow),
            method='Nelder-Mead',
            options={'maxiter': 500}  # Reduced for speed
        )

        if np.all(np.isfinite(result.x)) and np.all(result.x > 0):
            Lb = result.x[0]
            Wb = result.x[1]
            Kb = result.x[2]
            Kz = result.x[3]

            X = np.array([Lb, Wb, ALPHA, Ks])
            LOGX = np.log10(X)

            result = minimize(
                cal_surface,
                LOGX,
                args=(tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow),
                method='Nelder-Mead',
                options={'maxiter': 500}  # Reduced for speed
            )

            if np.all(np.isfinite(result.x)):
                Lb = 10 ** result.x[0]
                Wb = 10 ** result.x[1]
                ALPHA = 10 ** result.x[2]
                Ks = 10 ** result.x[3]
                
                # Update X1 to match new ALPHA value (X1 = 1/ALPHA)
                X1 = 1.0 / ALPHA
                
                basin_char = [tmp_area, Lb, X1, Wb, POR]
                gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

                SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                               gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
                bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)
                Error = calculate_error(bfs_out)

                tmp_bf = bfs_out[baseflow_col].copy()
                tmp_ov = bfs_out[baseflow_col] > bfs_out[qob_col]
                tmp_ov[np.isnan(tmp_ov)] = False
                tmp_bf[tmp_ov] = bfs_out[qob_col][tmp_ov]
                BFF = np.nansum(tmp_bf[~np.isnan(bfs_out[qob_col])]) / np.nansum(bfs_out[qob_col][~np.isnan(bfs_out[qob_col])])

                bf_params = pd.concat([bf_params, pd.DataFrame({
                    'tmp.site': [tmp_site],
                    'tmp.area': [tmp_area],
                    'Lb': [Lb],
                    'X1': [X1],
                    'Wb': [Wb],
                    'POR': [POR],
                    'ALPHA': [ALPHA],
                    'BETA': [BETA],
                    'Ks': [Ks],
                    'Kb': [Kb],
                    'Kz': [Kz],
                    'Qthresh': [Qthresh],
                    'Rs': [Rs],
                    'Rb1': [Rb1],
                    'Rb2': [Rb2],
                    'Prec': [Prec],
                    'Frac4Rise': [Frac4Rise],
                    'Error': [Error],
                    'BFF': [BFF]
                })], ignore_index=True)

    # STEP 3: SELECT BEST PARAMETERS TO MAXIMIZE BASEFLOW AND RE-CALIBRATE SURFACE PARAMETERS
    print("  Step 3: Final calibration with best parameters...")
    if len(bf_params) > 1:
        # Debug: Show all rows in bf_params
        print(f"    bf_params has {len(bf_params)} rows:")
        for idx in range(len(bf_params)):
            print(f"      Row {idx}: BETA={bf_params['BETA'].iloc[idx]:.3f}, BFF={bf_params['BFF'].iloc[idx]:.6f}, Error={bf_params['Error'].iloc[idx]:.6f}")
        
        # Select row with max BFF (excluding first row, matching R: bf_params$BFF[-1])
        g = bf_params['BFF'].iloc[1:].idxmax()
        Lb = bf_params['Lb'].iloc[g]
        X1 = bf_params['X1'].iloc[g]
        Wb = bf_params['Wb'].iloc[g]
        basin_char = [tmp_area, Lb, X1, Wb, POR]

        BETA = bf_params['BETA'].iloc[g]
        print(f"    Selected row {g}: BETA={BETA:.3f}, BFF={bf_params['BFF'].iloc[g]:.6f}")
        Kb = bf_params['Kb'].iloc[g]
        Ks = bf_params['Ks'].iloc[g]
        Kz = bf_params['Kz'].iloc[g]
        gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

        X = np.array([Lb, Wb, ALPHA, Ks, Kb, Kz])
        LOGX = np.log10(X)

        result = minimize(
            cal_initial,
            LOGX,
            args=(tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean),
            method='Nelder-Mead',
            options={'maxiter': 500}  # Reduced for speed
        )

        if np.all(np.isfinite(result.x)):
            Lb = 10 ** result.x[0]
            Wb = 10 ** result.x[1]
            ALPHA = 10 ** result.x[2]
            Ks = 10 ** result.x[3]
            Kb = 10 ** result.x[4]
            Kz = 10 ** result.x[5]
            
            # Update X1 to match new ALPHA value (X1 = 1/ALPHA)
            X1 = 1.0 / ALPHA
            
            basin_char = [tmp_area, Lb, X1, Wb, POR]
            gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

    # FINAL RUN
    print("  Running final BFS simulation...")
    SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                   gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
    bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)
    Error = calculate_error(bfs_out)

    tmp_bf = bfs_out[baseflow_col].copy()
    tmp_ov = bfs_out[baseflow_col] > bfs_out[qob_col]
    tmp_ov[np.isnan(tmp_ov)] = False
    tmp_bf[tmp_ov] = bfs_out[qob_col][tmp_ov]
    BFF = np.nansum(tmp_bf[~np.isnan(bfs_out[qob_col])]) / np.nansum(bfs_out[qob_col][~np.isnan(bfs_out[qob_col])])

    bf_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Lb': [np.round(Lb, 6)],
        'X1': [np.round(X1, 6)],
        'Wb': [np.round(Wb, 6)],
        'POR': [np.round(POR, 6)],
        'ALPHA': [np.round(ALPHA, 6)],
        'BETA': [np.round(BETA, 6)],
        'Ks': [np.round(Ks, 6)],
        'Kb': [np.round(Kb, 6)],
        'Kz': [np.round(Kz, 6)],
        'Qthresh': [np.round(Qthresh, 6)],
        'Rs': [np.round(Rs, 6)],
        'Rb1': [np.round(Rb1, 6)],
        'Rb2': [np.round(Rb2, 6)],
        'Prec': [np.round(Prec, 6)],
        'Frac4Rise': [np.round(Frac4Rise, 6)],
        'Error': [np.round(Error, 6)],
        'BFF': [np.round(BFF, 6)]
    })

    # Generate credible interval table
    tmp_ci = bf_ci(bfs_out)
    ci_table = tmp_ci[0]

    # Calculate the components as fractions of streamflow
    qsim_col = 'Qsim' if 'Qsim' in bfs_out.columns else 'Qsim.L3'
    surfaceflow_col = 'SurfaceFlow' if 'SurfaceFlow' in bfs_out.columns else 'SurfaceFlow.L3'
    directrunoff_col = 'DirectRunoff' if 'DirectRunoff' in bfs_out.columns else 'DirectRunoff.L3'

    tmp = (bfs_out[qob_col] > 0)
    tmp[np.isnan(tmp)] = False
    tmp[np.isnan(bfs_out[qsim_col])] = False
    bfs_out_filtered = bfs_out[tmp].copy()

    bff = pd.DataFrame({
        'tmp.site': [tmp_site],
        'Qmean': [np.round(np.nanmean(bfs_out_filtered[qob_col]), 6)],
        'BFF': [np.nan],
        'SFF': [np.nan],
        'DRF': [np.nan],
        'Error': [Error]
    })

    # Daily baseflow as fraction of observed streamflow
    tmp_bf = bfs_out_filtered[baseflow_col] / bfs_out_filtered[qob_col]
    tmp_bf[tmp_bf > 1] = 1

    # Daily surface flow as fraction of observed streamflow
    tmp_sf = bfs_out_filtered[surfaceflow_col] / bfs_out_filtered[qob_col]
    tmp_sf[(tmp_sf + tmp_bf) > 0] = 1 - tmp_bf[(tmp_sf + tmp_bf) > 0]

    # Mean baseflow, surface flow, and direct runoff fractions
    bff['BFF'] = np.round(np.nansum(tmp_bf * bfs_out_filtered[qob_col]) / np.nansum(bfs_out_filtered[qob_col]), 3)
    bff['SFF'] = np.round(np.nansum(tmp_sf * bfs_out_filtered[qob_col]) / np.nansum(bfs_out_filtered[qob_col]), 3)
    bff['DRF'] = np.round(np.nansum(bfs_out_filtered[directrunoff_col]) / np.nansum(bfs_out_filtered[qob_col]), 3)

    return (bf_params, bff, ci_table, bfs_out)

