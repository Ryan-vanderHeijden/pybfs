# Bug fix: SBT step-lookup causes baseflow freeze at low groundwater storage

## Summary

During long recession periods, the BFS model could enter a state where simulated
baseflow and total flow snapped to zero and remained frozen for weeks, even though
the groundwater store (`StBase`) still held significant water. This was caused by
a step-down discontinuity in the baseflow table (SBT) lookup.

## Root cause

The SBT is a lookup table mapping groundwater storage (`S`) to baseflow rate (`Q`),
groundwater-level elevation (`Z`), and the longitudinal position of the water-table
intersection with the surface (`Xb`). Its first row is always `S=0, Q=0, Xb=0`.

All four SBT lookups in the core loop used a **lower-bound index** — they returned
the value at the largest SBT row whose key did not exceed the query value. When
`StBase` fell below the second SBT row (the first non-zero entry), the lookup
returned index 0, mapping the current state to `Q=0` and `Xb=0`.

With `qb_in = 0` at the next time step:
- `sb_en = max(StBase + recharge - qb_in, 0)` → `StBase` stops draining
- `qcomp[:, 1] = (qb_in + qb_en) / 2 = 0` → baseflow = 0
- The model was permanently frozen until the next rise event

**Example (site 01134500):** `StBase` reached 25,102 m³ during a February 2004
recession. The first non-zero SBT entry was at `S = 25,537 m³`. The 435 m³ gap
was enough to trigger the snap, zeroing out 12 consecutive days of simulated flow.

## Fix

All SBT lookups were changed from lower-bound indexing to **linear interpolation**
between neighbouring table entries.

### JIT path (`_bfs_core_loop` in `bfs.py`)

A Numba-compatible helper was added:

```python
@jit(nopython=True, cache=True)
def _sbt_interp(key, key_arr, val_arr):
    n = len(key_arr)
    if key <= key_arr[0]:
        return val_arr[0]
    if key >= key_arr[n - 1]:
        return val_arr[n - 1]
    for i in range(n - 1):
        if key_arr[i] <= key <= key_arr[i + 1]:
            t = (key - key_arr[i]) / (key_arr[i + 1] - key_arr[i])
            return val_arr[i] + t * (val_arr[i + 1] - val_arr[i])
    return val_arr[n - 1]
```

Four lookups in `_bfs_core_loop` were updated:

| Purpose | Old | New |
|---|---|---|
| `xb_in → qb_in` | step index into `sbt_q` | `_sbt_interp(xb_in, sbt_xb, sbt_q)` |
| `sb_en → xb_en, zb_en, qb_en` | step index into each column | `_sbt_interp(sb_en, sbt_s, ...)` |
| `ST[ts,1] → Z[ts,1]` | step index into `sbt_z` | `_sbt_interp(ST[ts,1], sbt_s, sbt_z)` |
| `ST[ts,1] → X[ts]` | step index into `sbt_xb` | `_sbt_interp(ST[ts,1], sbt_s, sbt_xb)` |

### Python fallback path (`bfs.py`)

The equivalent four step-lookups in the Python `while` loop were replaced with
`np.interp` calls, e.g.:

```python
# Before
idx = max((SBT["S"] < sb_en).sum(), 1) - 1
qb_en = SBT["Q"].iloc[idx]

# After
qb_en = np.interp(sb_en, SBT["S"].values, SBT["Q"].values)
```

## Result

The 12 frozen days in the February 2004 recession (site 01134500) were resolved.
`StBase` now drains continuously through the low-storage period, and simulated
baseflow remains physically plausible throughout the record.

---

# Bug fix: `rech_en` uses `xb_in` instead of `xb_en`, underestimating recharge during recession onset

## Summary

Even after the SBT interpolation fix, simulated baseflow still dropped too low
during extended dry periods. The GW store was entering dry spells slightly
under-filled due to underestimated end-of-step recharge during wet-to-dry
transitions.

## Root cause

End-of-step recharge is computed as:

```python
rech_en = min(recharge(lb, xb_in, ws, kz, zs_en, por), sba + qb_in)
```

The `recharge` formula is `(lb - xb) * 2 * ws * min(zs * por, kz)`, where
`(lb - xb)` is the unsaturated-zone length that receives recharge. As the GW
store drains during a recession, `xb` decreases — meaning the recharge area
*grows*. But by using `xb_in` (beginning-of-step value, which is higher), the
code under-counted the recharge area at the end of the step. The error was
largest during the early part of a recession when `xb` is dropping rapidly.

The effect: each storm event left the GW store slightly under-filled compared
to the physical expectation, so simulated baseflow reached near-zero sooner
than observed during long dry periods.

## Fix

A two-pass correction was applied in both `_bfs_core_loop` (JIT) and the
Python fallback in `bfs.py`:

1. **First pass** — compute a provisional `sb_en_prov` using `xb_in` (as before).
2. **Derive** `xb_en_prov` from `sb_en_prov` via SBT interpolation.
3. **Second pass** — recompute `rech_en` using the midpoint `(xb_in + xb_en_prov) / 2`,
   then compute the final `sb_en`.

```python
# First pass (provisional)
rech_en = min(recharge(lb, xb_in, ws, kz, zs_en, por), sba + qb_in)
sb_en_prov = max(sb_in + rech_en - qb_in, 0)
xb_en_prov = interp(sb_en_prov, sbt_s, sbt_xb)

# Second pass (midpoint Xb)
rech_en = min(recharge(lb, (xb_in + xb_en_prov) / 2, ws, kz, zs_en, por), sba + qb_in)
sb_en = max(sb_in + rech_en - qb_in, 0)
```

This is equivalent to one step of a trapezoidal (Heun's method) correction for
the Xb-dependent recharge term, and avoids the circular dependency between
`rech_en` and `xb_en`.

Note: `rech_in` (beginning-of-step recharge) correctly uses `xb_in` and
requires no change.
