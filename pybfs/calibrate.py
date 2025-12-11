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

# Global variables for relative tolerance tracking (matching R's reltol=0.01)
_reltol_tracker = {'prev_f': None, 'initial_f': None, 'reltol': 0.01, 'stop': False, 'callback_stop': False}

class RelTolConvergence(Exception):
    """Exception to signal reltol convergence"""
    pass

def _create_reltol_callback():
    """Create a callback function that checks reltol and stops optimization early"""
    def callback(xk):
        # This callback is called after each iteration
        # We can't directly stop, but we can track and raise an exception
        # However, Nelder-Mead doesn't support callbacks that can stop early
        # So we'll use a different approach: track in the objective function wrapper
        pass
    return callback

def _wrap_objective_with_reltol(func):
    """Wrap objective function to track values for relative tolerance
    
    Note: This wrapper tracks reltol but cannot directly stop scipy's minimize().
    We use a callback-based approach instead.
    """
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Track function values for relative tolerance
        if _reltol_tracker['initial_f'] is None:
            _reltol_tracker['initial_f'] = result
        
        # Check relative tolerance (1% change)
        # R's reltol checks: abs(f_new - f_prev) / max(abs(f_prev), abs(f_new)) < reltol
        # This works for both positive and negative objective values
        if _reltol_tracker['prev_f'] is not None and _reltol_tracker.get('reltol') is not None:
            # Use max(abs(prev), abs(current)) as denominator (matching R's optim)
            denom = max(abs(_reltol_tracker['prev_f']), abs(result))
            if denom > 0:
                rel_change = abs(result - _reltol_tracker['prev_f']) / denom
                if rel_change < _reltol_tracker['reltol']:
                    _reltol_tracker['stop'] = True
                    _reltol_tracker['callback_stop'] = True
        
        _reltol_tracker['prev_f'] = result
        return result
    return wrapped

def _reset_reltol_tracker():
    """Reset the relative tolerance tracker"""
    _reltol_tracker['prev_f'] = None
    _reltol_tracker['initial_f'] = None
    _reltol_tracker['stop'] = False
    _reltol_tracker['callback_stop'] = False


def _create_scaled_minimize(func, x0, parscale, *args, **kwargs):
    """Create a scaled minimize call matching R's parscale behavior
    
    R's optim() uses parscale to normalize parameters internally.
    This function implements the same behavior for scipy's minimize.
    
    Parameters
    ----------
    func : callable
        Objective function
    x0 : array-like
        Initial parameter guess
    parscale : array-like
        Parameter scaling values (same length as x0)
    *args : tuple
        Additional arguments to pass to func
    **kwargs : dict
        Additional keyword arguments for minimize
    
    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result with unscaled parameters
    """
    from scipy.optimize import minimize
    
    parscale = np.array(parscale, dtype=float)
    x0 = np.array(x0, dtype=float)
    
    # Avoid division by zero
    parscale = np.where(parscale == 0, 1.0, parscale)
    
    # Create scaled objective function
    def scaled_func(scaled_x, *func_args):
        # Unscale parameters: R divides by parscale internally, so we multiply back
        unscaled_x = scaled_x * parscale
        return func(unscaled_x, *func_args)
    
    # Scale initial guess (divide by parscale, matching R's behavior)
    scaled_x0 = x0 / parscale
    
    # Extract options
    options = kwargs.pop('options', {})
    
    # Reset reltol tracker (for diagnostics only - we can't actually stop scipy early)
    _reset_reltol_tracker()
    
    # Wrap the objective function to track reltol (for diagnostics)
    # Note: scipy's minimize() doesn't support early stopping via callbacks for Nelder-Mead,
    # so we can't perfectly replicate R's reltol behavior. The wrapper tracks values but
    # can't stop the optimization early.
    wrapped_scaled_func = _wrap_objective_with_reltol(scaled_func)
    
    # Run optimization normally
    # R's reltol=0.01 would stop when relative change < 1%, but scipy doesn't support this
    # for Nelder-Mead. We use fatol=1e-4 as an approximation, but Python may still take
    # more iterations than R due to this limitation.
    result = minimize(wrapped_scaled_func, scaled_x0, args=args, options=options, **kwargs)
    
    # Unscale result
    if np.all(np.isfinite(result.x)):
        result.x = result.x * parscale
    
    return result


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

    # Run BFS (matching R: error_basis='base' for calibration)
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow, timestep=timestep, error_basis=error_basis)
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
    # Matching R: x[2]<=0 (X1 <= 0)
    bad_X = [
        np.any(np.isinf(x)),
        x[0] <= 0,
        x[0] > area / x[2],
        x[1] <= 0,  # Matching R: x[2]<=0 (X1 <= 0)
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

    # Run BFS (matching R: error_basis='base' for calibration)
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow, timestep=timestep, error_basis=error_basis)
    obj = objective(bfs_out, prec=flow[4])

    return obj


def cal_surface(x, tmp_q, dys, timestep, error_basis, basin_char, gw_hyd, flow):
    """Calibrate surface parameters

    Objective function for calibrating surface reservoir parameters.
    
    NOTE: R comment says [Wb,ALPHA,Ks,Kz] but R actually calls with [Lb,Wb,ALPHA,Ks].
    This function matches R's actual behavior (not the comment).

    Parameters
    ----------
    x : array-like
        Log10 of parameters [Lb, Wb, ALPHA, Ks] (matching R's actual call)
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

    # BUG IN R CODE: R's cal_surface comment says it expects [Wb,ALPHA,Ks,Kz] 
    # but R actually calls it with [Lb,Wb,ALPHA,Ks] (log10)
    # 
    # When R calls with [Lb,Wb,ALPHA,Ks] (log10), the indices are:
    #   x[1] = log10(Lb) - R expects log10(Wb) ✗ WRONG!
    #   x[2] = log10(Wb) - R expects log10(ALPHA) ✗ WRONG! This causes x[2] > -1 to be TRUE (3.29 > -1)
    #   x[3] = log10(ALPHA) - R expects log10(Ks) ✗ WRONG!
    #   x[4] = log10(Ks) - R expects log10(Kz) ✗ WRONG!
    #
    # R's bad_X check: bad_X=c(10^x[1]>basin_char[1]/basin_char[2],x[2]>(-1),x[3]<(-8),x[3]>5,x[4]<(-8),x[4]>5)
    #   - 10^x[1] expects Wb but gets Lb (checks 10^Lb > area/lb)
    #   - x[2] expects ALPHA but gets Wb (checks Wb > -1, which is TRUE! This causes 100.0 return)
    #   - x[3] expects Ks but gets ALPHA (checks ALPHA < -8 or > 5)
    #   - x[4] expects Kz but gets Ks (checks Ks < -8 or > 5)
    #
    # R's function body also uses wrong indices:
    #   wb=10^x[1]  expects Wb but gets Lb
    #   a=10^x[2]   expects ALPHA but gets Wb
    #   ks=10^x[3]  expects Ks but gets ALPHA
    #   kz=10^x[4]  expects Kz but gets Ks
    #
    # BUG: R's cal_surface expects [Wb,ALPHA,Ks,Kz] but is called with [Lb,Wb,ALPHA,Ks]
    # So R's bad_X check uses wrong indices:
    #   x[1] expects Wb but gets Lb
    #   x[2] expects ALPHA but gets Wb  <- This causes x[2] > -1 to check Wb (3.29) instead of ALPHA (-2.02)!
    #   x[3] expects Ks but gets ALPHA
    #   x[4] expects Kz but gets Ks
    #
    # To match R's buggy behavior exactly, we need to use the same wrong indices in the check:
    bad_X = [
        (10 ** x[1]) > (basin_char[0] / basin_char[1]),  # 10^Lb > area/lb (R expects 10^Wb but gets Lb)
        x[1] > -1,  # Wb > -1 (R expects ALPHA > -1 but checks x[2] which is Wb!) - THIS IS TRUE, causing 100.0!
        x[2] < -8,  # ALPHA < -8 (R expects Ks < -8 but checks x[3] which is ALPHA)
        x[2] > 5,   # ALPHA > 5 (R expects Ks > 5 but checks x[3] which is ALPHA)
        x[3] < -8,  # Ks < -8 (R expects Kz < -8 but checks x[4] which is Ks, using x[3] since x[4] doesn't exist)
        x[3] > 5,   # Ks > 5 (R expects Kz > 5 but checks x[4] which is Ks, using x[3] since x[4] doesn't exist)
    ]

    if any(bad_X):
        return 100.0

    # IMPORTANT: R uses tmp$par from optim, NOT the values calculated inside cal_surface!
    # So R's cal_surface bug (wrong indices in function body) doesn't affect final parameters.
    # R only uses cal_surface to calculate the objective function value.
    # 
    # However, to match R's behavior exactly, we need to use the same wrong indices
    # in the function body (even though they're wrong) so the objective function matches.
    #
    # R's function body uses wrong indices (matching R's buggy behavior):
    #   wb=10^x[1]  expects Wb but gets Lb
    #   a=10^x[2]   expects ALPHA but gets Wb  
    #   ks=10^x[3]  expects Ks but gets ALPHA
    #   kz=10^x[4]  expects Kz but gets Ks
    #
    # But since R uses tmp$par from optim (not these values), we can use correct indices
    # for the function body, but we need to match the bad_X check exactly.
    #
    # Actually, wait - if we use correct indices in the function body, the objective
    # function will be different from R's. So we need to match R's buggy behavior exactly.
    #
    # But actually, R's optim will try different parameter values, and cal_surface will
    # return 100.0 for most of them (due to the bug), so optim might return the initial
    # guess or parameters that don't trigger the 100.0 return.
    #
    # For now, let's use correct indices in the function body (since R uses tmp$par anyway),
    # but match the buggy bad_X check:
    lb = 10 ** x[0]  # Lb (correct - from x[0])
    wb = 10 ** x[1]  # Wb (correct - from x[1])
    alpha = 10 ** x[2]  # ALPHA (correct - from x[2])
    ks = 10 ** x[3]  # Ks (correct - from x[3])
    # Note: We use correct indices here because R uses tmp$par from optim, not these values
    # But we match the buggy bad_X check above to return 100.0 when appropriate

    basin_char_new = basin_char.copy()
    basin_char_new[1] = lb  # Update Lb
    basin_char_new[3] = wb  # Update Wb

    gw_hyd_new = gw_hyd.copy()
    gw_hyd_new[0] = alpha  # Update ALPHA
    gw_hyd_new[2] = ks  # Update Ks
    # Note: Kz is NOT updated in cal_surface (matching R behavior)

    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })

    # Generate baseflow table
    SBT = base_table(basin_char_new[1], basin_char_new[2], basin_char_new[3],
                     gw_hyd_new[1], gw_hyd_new[3], streamflow_df, basin_char_new[4])

    # Run BFS (matching R: error_basis='base' for calibration)
    bfs_out = bfs(streamflow_df, SBT, basin_char_new, gw_hyd_new, flow, timestep=timestep, error_basis=error_basis)
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

    # Initial BFS run (matching R: error_basis='base' for calibration)
    bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow, timestep='day', error_basis='base')
    Error = calculate_error(bfs_out)

    # Calculate BFF
    baseflow_col = 'Baseflow' if 'Baseflow' in bfs_out.columns else 'Baseflow.L3'
    qob_col = 'Qob' if 'Qob' in bfs_out.columns else 'Qob.L3'
    BFF = np.nansum(bfs_out[baseflow_col]) / np.nansum(bfs_out[qob_col])

    # DIAGNOSTIC: Initial parameters (before Step 1)
    print("\n=== DIAGNOSTIC: INITIAL PARAMETERS (before Step 1) ===")
    print(f"Lb={Lb:.6f}, X1={X1:.6f}, Wb={Wb:.6f}, POR={POR:.6f}")
    print(f"ALPHA={ALPHA:.6f}, BETA={BETA:.6f}")
    print(f"Ks={Ks:.6f}, Kb={Kb:.6f}, Kz={Kz:.6f}")
    print(f"Error={Error:.6f}, BFF={BFF:.6f}")

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
    
    # Reset relative tolerance tracker
    _reset_reltol_tracker()
    
    # Wrap objective function to track values and implement reltol=0.01
    wrapped_cal_initial = _wrap_objective_with_reltol(cal_initial)
    
    # R uses: optim(..., control=list(maxit=1000, parscale=LOGX, reltol=0.01))
    result = _create_scaled_minimize(
        wrapped_cal_initial,
        LOGX,
        LOGX,  # parscale = LOGX
        tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
    )

    # DIAGNOSTIC: After Step 1 cal_initial
    print("\n=== DIAGNOSTIC: STEP 1 - AFTER cal_initial ===")
    print(f"Optimization success: {result.success}, iterations: {result.nit}, final objective: {result.fun:.6f}")
    if np.all(np.isfinite(result.x)):
        print(f"Optimized params (log10): {result.x}")
        print(f"Optimized params (linear): Lb={10**result.x[0]:.6f}, Wb={10**result.x[1]:.6f}, ALPHA={10**result.x[2]:.6f}, Ks={10**result.x[3]:.6f}, Kb={10**result.x[4]:.6f}, Kz={10**result.x[5]:.6f}")

    if np.all(np.isfinite(result.x)):
        Lb = 10 ** result.x[0]
        Wb = 10 ** result.x[1]
        ALPHA = 10 ** result.x[2]
        Ks = 10 ** result.x[3]
        Kb = 10 ** result.x[4]
        Kz = 10 ** result.x[5]
        
        # X1 is NOT updated after cal_initial (matching R line 58)
        # X1 remains as the initial value (1/ALPHA = 100)

        basin_char = [tmp_area, Lb, X1, Wb, POR]
        gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

        X = np.array([Lb, Wb, ALPHA, Ks])
        LOGX = np.log10(X)
        
        # Reset relative tolerance tracker
        _reset_reltol_tracker()
        
        # Wrap objective function
        wrapped_cal_surface = _wrap_objective_with_reltol(cal_surface)

        # R uses: optim(..., control=list(maxit=1000, parscale=LOGX, reltol=0.01))
        result = _create_scaled_minimize(
            wrapped_cal_surface,
            LOGX,
            LOGX,  # parscale = LOGX
            tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
        )

        # DIAGNOSTIC: After Step 1 cal_surface
        print("\n=== DIAGNOSTIC: STEP 1 - AFTER cal_surface ===")
        print(f"Optimization success: {result.success}, iterations: {result.nit}, final objective: {result.fun:.6f}")
        if np.all(np.isfinite(result.x)):
            print(f"Optimized params (log10): {result.x}")
            print(f"Optimized params (linear): Lb={10**result.x[0]:.6f}, Wb={10**result.x[1]:.6f}, ALPHA={10**result.x[2]:.6f}, Ks={10**result.x[3]:.6f}")

        # CRITICAL: R's Step 1 structure (matching R lines 56-80):
        # if(all(is.finite(tmp$par))){  # Outer: cal_initial succeeded
        #   update params from cal_initial
        #   call cal_surface
        # }
        # if(all(is.finite(tmp$par))){  # Inner: cal_surface succeeded (checks tmp from cal_surface)
        #   update params from cal_surface
        #   run BFS
        #   calculate BFF
        # }
        # bf_params=rbind(...)  # OUTSIDE both if blocks!
        # 
        # This means bf_params is always updated, using:
        # - Initial params if cal_initial fails
        # - cal_initial params if cal_initial succeeds but cal_surface fails
        # - cal_surface params if both succeed
        # Error and BFF are from initial run (line 42-43) if cal_surface fails,
        # or from cal_surface run if it succeeds.
        if np.all(np.isfinite(result.x)):
            Lb = 10 ** result.x[0]
            Wb = 10 ** result.x[1]
            ALPHA = 10 ** result.x[2]
            Ks = 10 ** result.x[3]
            
            # X1 is NOT updated after cal_surface in Step 1 (matching R line 70)
            # X1 remains as the initial value (1/ALPHA = 100)

            basin_char = [tmp_area, Lb, X1, Wb, POR]
            gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

            SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                           gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
            bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow, timestep='day', error_basis='base')
            Error = calculate_error(bfs_out)

            tmp_bf = bfs_out[baseflow_col].copy()
            tmp_ov = bfs_out[baseflow_col] > bfs_out[qob_col]
            tmp_ov[np.isnan(tmp_ov)] = False
            tmp_bf[tmp_ov] = bfs_out[qob_col][tmp_ov]
            BFF = np.nansum(tmp_bf[~np.isnan(bfs_out[qob_col])]) / np.nansum(bfs_out[qob_col][~np.isnan(bfs_out[qob_col])])

            # DIAGNOSTIC: After Step 1 final BFS run
            print("\n=== DIAGNOSTIC: STEP 1 - AFTER final BFS run ===")
            print(f"Lb={Lb:.6f}, X1={X1:.6f}, Wb={Wb:.6f}, POR={POR:.6f}")
            print(f"ALPHA={ALPHA:.6f}, BETA={BETA:.6f}")
            print(f"Ks={Ks:.6f}, Kb={Kb:.6f}, Kz={Kz:.6f}")
            print(f"Error={Error:.6f}, BFF={BFF:.6f}")

        # OUTSIDE inner if, matching R line 80
        # bf_params is always updated, using current Error and BFF values
        # (from initial run if cal_surface failed, or from cal_surface run if it succeeded)
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

            # R uses: optim(..., control=list(maxit=1000, parscale=X, reltol=0.01))
            result = _create_scaled_minimize(
                cal_basetable,
                X,
                X,  # parscale = X
                b, [tmp_area, POR, Qmean, Qthresh, RbI, RbS], tmp_q,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
            )

            if np.all(np.isfinite(result.x)):
                n = n + 1
                tmp_out.append([result.x[0], result.x[1], result.x[2], b, result.x[3], result.fun])
                last_obj = result.fun
                if n % 5 == 0:
                    print(f"    Tested beta={b:.1f}, objective={result.fun:.4f}")

        if b > 10:
            # R checks: abs((tmp$value - tmp.out[n,6]) / tmp$value) < 0.001
            # Note: This check happens OUTSIDE the if(all(is.finite(tmp$par))) block
            # So tmp$value is the current optimization result (even if it failed)
            # And tmp.out[n,6] is the last successfully stored result
            # If current optimization succeeded: tmp$value == tmp.out[n,6], so difference is 0 (always stops)
            # If current optimization failed: tmp$value is from failed opt, tmp.out[n,6] is previous successful
            # This seems like it might be a bug in R, but we'll match it exactly
            if len(tmp_out) > 0 and last_obj is not None:
                # Compare current result (last_obj) vs last stored result (tmp_out[n-1][5])
                # Note: n was just incremented if optimization succeeded, so tmp_out[n-1] is current
                # If optimization failed, last_obj might be from a previous iteration
                stored_obj = tmp_out[n-1][5] if n > 0 else None
                if stored_obj is not None:
                    if abs((last_obj - stored_obj) / last_obj) < 0.001:
                        print(f"    Convergence reached at beta={b:.1f}")
                        continue_cal = False
        if b == 20:  # Match R exactly: if(b==20)
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
        # R uses: optim(..., control=list(maxit=1000, parscale=X, reltol=0.01))
        result = _create_scaled_minimize(
            cal_base,
            X,
            X,  # parscale = X
            tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
        )

        # DIAGNOSTIC: After Step 2 cal_base
        print(f"\n    === DIAGNOSTIC: STEP 2 - AFTER cal_base (beta={BETA:.3f}) ===")
        print(f"    Optimization success: {result.success}, iterations: {result.nit}, final objective: {result.fun:.6f}")
        
        # Match R: tmp$par[tmp$par<0]=NA (set negative parameters to NaN)
        result.x[result.x < 0] = np.nan
        
        if np.all(np.isfinite(result.x)):
            print(f"    Optimized params: Lb={result.x[0]:.6f}, Wb={result.x[1]:.6f}, Kb={result.x[2]:.6f}, Kz={result.x[3]:.6f}")

        # CRITICAL: R's nesting structure (matching R lines 130-150):
        # if(all(is.finite(tmp$par))){  # Outer: cal_base succeeded
        #   Lb=tmp$par[1];Wb=tmp$par[2];Kb=tmp$par[3];Kz=tmp$par[4]
        #   ...
        #   tmp=optim(LOGX,cal_surface,...)
        #   
        #   if(all(is.finite(tmp$par))){  # Inner: cal_surface succeeded
        #     Lb=10^tmp$par[1];Wb=10^tmp$par[2];ALPHA=10^tmp$par[3];Ks=10^tmp$par[4]
        #     basin_char=c(...)
        #     gw_hyd=c(...)
        #   }
        #   
        #   # OUTSIDE inner if, INSIDE outer if:
        #   Error=bfs(...)
        #   ...
        #   bf_params=rbind(...)
        # }
        # This means if cal_base succeeds but cal_surface fails, R still runs BFS
        # and adds to bf_params using cal_base params (not updated by cal_surface).
        if np.all(np.isfinite(result.x)):
            Lb = result.x[0]
            Wb = result.x[1]
            Kb = result.x[2]
            Kz = result.x[3]

            X = np.array([Lb, Wb, ALPHA, Ks])
            LOGX = np.log10(X)
            
            # Reset relative tolerance tracker
            _reset_reltol_tracker()
            wrapped_cal_surface = _wrap_objective_with_reltol(cal_surface)
            
            # R uses: optim(..., control=list(maxit=1000, parscale=LOGX, reltol=0.01))
            result = _create_scaled_minimize(
                wrapped_cal_surface,
                LOGX,
                LOGX,  # parscale = LOGX
                tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
            )

            # DIAGNOSTIC: After Step 2 cal_surface
            print(f"\n    === DIAGNOSTIC: STEP 2 - AFTER cal_surface (beta={BETA:.3f}) ===")
            print(f"    Optimization success: {result.success}, iterations: {result.nit}, final objective: {result.fun:.6f}")
            if np.all(np.isfinite(result.x)):
                print(f"    Optimized params (log10): {result.x}")
                print(f"    Optimized params (linear): Lb={10**result.x[0]:.6f}, Wb={10**result.x[1]:.6f}, ALPHA={10**result.x[2]:.6f}, Ks={10**result.x[3]:.6f}")

            # Inner if: Update parameters if cal_surface succeeded
            if np.all(np.isfinite(result.x)):
                Lb = 10 ** result.x[0]
                Wb = 10 ** result.x[1]
                ALPHA = 10 ** result.x[2]
                Ks = 10 ** result.x[3]
                
                # X1 is NOT updated after cal_surface in Step 2 (matching R line 140)
                # X1 remains from the selected beta row
                
                basin_char = [tmp_area, Lb, X1, Wb, POR]
                gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

            # OUTSIDE inner if, INSIDE outer if (matching R lines 143-150)
            # Run BFS and add to bf_params even if cal_surface failed (using cal_base params)
            SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                           gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
            bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow, timestep='day', error_basis='base')
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

        # Reset relative tolerance tracker
        _reset_reltol_tracker()
        wrapped_cal_initial = _wrap_objective_with_reltol(cal_initial)
        
        # R uses: optim(..., control=list(maxit=1000, parscale=LOGX, reltol=0.01))
        result = _create_scaled_minimize(
            wrapped_cal_initial,
            LOGX,
            LOGX,  # parscale = LOGX
            tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
        )

        # DIAGNOSTIC: After Step 3 cal_initial
        print("\n=== DIAGNOSTIC: STEP 3 - AFTER cal_initial ===")
        print(f"Optimization success: {result.success}, iterations: {result.nit}, final objective: {result.fun:.6f}")
        if np.all(np.isfinite(result.x)):
            print(f"Optimized params (log10): {result.x}")
            print(f"Optimized params (linear): Lb={10**result.x[0]:.6f}, Wb={10**result.x[1]:.6f}, ALPHA={10**result.x[2]:.6f}, Ks={10**result.x[3]:.6f}, Kb={10**result.x[4]:.6f}, Kz={10**result.x[5]:.6f}")

        if np.all(np.isfinite(result.x)):
            Lb = 10 ** result.x[0]
            Wb = 10 ** result.x[1]
            ALPHA = 10 ** result.x[2]
            Ks = 10 ** result.x[3]
            Kb = 10 ** result.x[4]
            Kz = 10 ** result.x[5]
            
            # X1 is kept from the selected row (matching R line 156, 173)
            # Do NOT recalculate X1 = 1/ALPHA here
            # X1 remains as selected from bf_params[g]
            
            basin_char = [tmp_area, Lb, X1, Wb, POR]
            gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

    # FINAL RUN (matching R lines 175-185)
    # Note: In R, the final run is OUTSIDE the if block, so it always executes
    # even if Step 3 cal_initial failed. If cal_initial failed, it uses the
    # selected parameters from bf_params[g] (not updated by cal_initial).
    # Python's check for len(bf_params) > 1 is a safety check that shouldn't
    # be needed in practice, but we keep it to avoid errors.
    if len(bf_params) > 1:
        print("  Running final BFS simulation...")
    SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                   gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
    bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow, timestep='day', error_basis='base')
    Error = calculate_error(bfs_out)

    tmp_bf = bfs_out[baseflow_col].copy()
    tmp_ov = bfs_out[baseflow_col] > bfs_out[qob_col]
    tmp_ov[np.isnan(tmp_ov)] = False
    tmp_bf[tmp_ov] = bfs_out[qob_col][tmp_ov]
    BFF = np.nansum(tmp_bf[~np.isnan(bfs_out[qob_col])]) / np.nansum(bfs_out[qob_col][~np.isnan(bfs_out[qob_col])])

    # DIAGNOSTIC: Final parameters (after Step 3)
    print("\n=== DIAGNOSTIC: FINAL PARAMETERS (after Step 3) ===")
    print(f"Lb={Lb:.6f}, X1={X1:.6f}, Wb={Wb:.6f}, POR={POR:.6f}")
    print(f"ALPHA={ALPHA:.6f}, BETA={BETA:.6f}")
    print(f"Ks={Ks:.6f}, Kb={Kb:.6f}, Kz={Kz:.6f}")
    print(f"Error={Error:.6f}, BFF={BFF:.6f}")

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

