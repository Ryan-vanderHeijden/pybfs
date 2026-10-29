# Geometry kernels vs. the USGS BFS manual

Source: Konrad, C.P., 2022, *BFS—A non-linear, state-space model for baseflow
separation and prediction*, USGS SIR 2022-5114 (`docs/usgs_bfs_manual.pdf`,
untracked). Page numbers are **report** pages (PDF page = report page + 10).
Every equation below was read from a rendered page image, because the PDF text layer
scrambles the math. Code references are to `baseline-original` (`d8e2446`);
`R:` refers to `bfs/source/Rfunctions.bfs_utilities.R`.

**Version caveat:** the manual documents the R function `bf_sep` and cites its line
numbers. The R code in `bfs/source/` is a later revision (`bfs`, see
`bfs/RecalibrationDescription.txt`, which describes changes to errors, weights,
`flow_metrics`, and `base_table`). A code/manual difference can therefore be a
deliberate revision rather than an error. None of the revision notes mention the
six geometry kernels.

Rule followed: the geometry comes only from the manual's prose, Figure 1, and
Tables 1, 2, 4, 5. Where I compare a code formula against that geometry, the
geometry is the manual's statement, not a reverse-engineering of the code.

---

## 1. Symbols as the manual defines them

| Symbol | Manual definition (verbatim or near-verbatim) | Where | Code name(s) |
|---|---|---|---|
| Lb | "The basin has a length of Lb" (p. 4); "the width (perpendicular to flow) of surface reservoir" (eq. 9 key, p. 8); "Effective length of stream basin and channel, **width of base reservoir**" (table 4, p. 10) | p. 4, 8, 10 | `lb` |
| Wb | Basin width (fig. 1 caption); "the width of the base reservoir" (eq. 8 key, p. 8); "Effective base reservoir width" (table 4); "effective width, Wb, is perpendicular to the channel" (p. 10) | p. 4, 8, 10 | `wb` |
| Wb/2 | Not named as a symbol. Table 2 lists `Wb/2` as the argument of `sur_z`, `sur_store`, `infiltration`, `recharge`. Fig. 1 shows Wb spanning both hillslopes | p. 7, fig. 1 | `ws` |
| Zs(t) | "saturated thickness of the surface reservoir at the section where the water surface intersects the land surface" (eq. 9 key, p. 8; fig. 1 caption) | p. 4, 8 | `zs`, `z`, `Z[:,0]`, `Zs.L` |
| Xs(t) | Lateral distance from the channel to "the cross sections at Xs(t) on both sides of the channel where the water surface intersects the land surface" (p. 4); `Zs(Xs) = α Xs` (eq. 3) | p. 4, 5 | none (implicit as `zs/a`) |
| Ss(t) | Storage in the surface reservoir | p. 5 | `ss`, `ST[:,0]` |
| α | "the upper boundary has a slope, α, and the lower boundary is level with the channel" (p. 5); "Effective lateral hydraulic gradient of surface reservoir" (table 5); "The hydraulic gradient for discharge from the surface reservoir is equal to α" (p. 5) | p. 5, 11 | `alpha`, `a` |
| POR | "the drainable porosity" (eq. 4 key); "Effective drainable porosity" (table 4) | p. 5, 10 | `por` |
| Ks | "hydraulic conductivity for the surface reservoir" (eq. 9 key); table 5 use: "Infiltration into and discharge from surface reservoir" | p. 8, 11 | `ks` |
| Kz | "Effective vertical hydraulic conductivity of base reservoir"; use: "Recharge from surface reservoir into base reservoir" (table 5) | p. 11 | `kz` |
| Xb(t) | Distance from the outlet to "the cross section at Xb(t) where the water surface in the base reservoir intersects the channel"; "water-surface elevation is horizontal from Xb(t) to Lb" (p. 4) | p. 4 | `xb`, `X` |
| I(t) | Impulse of precipitation/snowmelt, a depth per time step (p. 4, eq. 13) | p. 4, 8 | `i`, `I`, `Impulse.L` |
| F(t) | Infiltration of the impulse, "only permitted through the unsaturated portion of the land surface" (p. 4) | p. 4, 8 | `infil`, `EXC[:,0]` |
| Qs(t), Qd(t), R(t) | Surface discharge, direct runoff, recharge (table 1) | p. 5 | `qs`, `qd`/`qcomp[:,2]`, `rech` |

**Geometry as the manual states it** (p. 4–5, fig. 1). The basin is a rectangular
"open book" dihedral, Lb long and Wb wide, with two symmetric hillslopes draining to
a channel along the center line. The surface reservoir has a triangular section
whose upper boundary rises at slope α from the channel and whose lower boundary
is level with the channel. Its thickness is a linear function of lateral distance,
`Zs(Xs) = α Xs` (eq. 3). Flow in it is lateral, governed by the sections at ±Xs. The
lower boundaries of both reservoirs are horizontal. Land surface between the
channel and Xs is saturated (direct runoff); beyond Xs it is unsaturated
(infiltration). The surface reservoir's flow width (perpendicular to its flow) is Lb.

---

## 2. Kernel by kernel

### 2.1 `sur_store(lb, a, ws, por, zs)`: surface storage from Zs

- **Manual:** eq. 4, p. 5: **Ss(t) = Xb(t) Zs(t) POR (Wb(t)/4 + Zs(t)/2α)**. Table 2: `sur_store(α, Wb/2, POR, Zs)`, "Calculates Ss given the parameters for equation 4 and Zs (inverse of sur_z)".
- **Code:** `utilities.py:59–89`, `R:73–75`: `Ss = Lb·POR·(2·ws·Zs − Zs²/α)` = `Lb·POR·Zs·(Wb − Zs/α)`.
- **Departures and ambiguities**
  1. **Length factor:** the manual has Xb(t), the base-reservoir intersect distance; the code has Lb. In the manual's own geometry the surface reservoir spans the whole channel (its flow width is Lb in eq. 9, and it recharges over Lb − Xb in eq. 16). Xb in eq. 4 therefore conflicts with the manual's other equations. **Ambiguous in the manual**; most likely a typesetting error for Lb.
  2. **Bracket:** `Zs(Wb/4 + Zs/2α)` vs. the code's `Zs(Wb − Zs/α)`. The Wb coefficient differs (¼ vs 1) and so does the sign of the Zs² term. The manual's stated geometry implies a storage that is *concave* in Zs: the saturated cross-section loses the triangle above the land surface between the channel and Xs, which gives a negative Zs² term. Eq. 4 as printed is convex. **I cannot reconcile eq. 4 with the manual's own geometry**, so I'm flagging it rather than deriving a "correct" form.
  3. `Wb(t)` is written as time-dependent; Wb is a constant parameter (table 4). This is a notation slip.
  4. Table 2's argument list has no length factor (neither Lb nor Xb). The R and Python functions take `lb`.
  5. `z = min(ws*a, zs)` is computed and unused (issue U6), so the code doesn't enforce the manual's implied cap Zs ≤ α·Wb/2. Callers cap instead (`bfs.py:184`).
- **Code vs. code:** the code's formula is internally consistent with eqs. 9, 15/19 and 16 (§2.3–2.6). It is **not** consistent with `sur_z` (next).

### 2.2 `sur_z(lb, a, ws, por, ss)`: Zs from surface storage

- **Manual:** table 2: `sur_z(α, Wb/2, POR, Ss)`, "Calculates the saturated thickness of the surface reservoir, Zs, given the parameters for equation 4 and the volume of water stored". The p. 5 prose instead attaches sur_z to **eq. 3** (`Zs(Xs) = α Xs`, "using the utility function sur_z"). The manual is internally inconsistent about which equation sur_z implements.
- **Code:** `utilities.py:18–56`, `R:78–80`: root of `a1·Zs² + b1·Zs + c1 = 0` with `a1 = 1/(2α)`, `b1 = −2ws`, `c1 = Ss/(Lb·POR)`; returns `ws·α` if the discriminant is negative.
- **Departures**
  1. **`sur_z` is not the inverse of `sur_store`** (verified numerically, 12167000 geometry). With `a1 = 1/(2α)` it inverts `Ss = Lb·POR·(2ws·Zs − Zs²/(2α))`, not `sur_store`'s `… − Zs²/α`. Round-trip `sur_z(sur_store(Zs))/Zs` is 0.997 at 1% of the maximum thickness, 0.91 at 30%, 0.84 at 50%, and 0.586 at full saturation. The R code has the same coefficient, so this is inherited. The Python docstring claims it "solves ss = lb·por·(2·ws·zs − zs²/a)", which is false. Logged as **M10** in `suspected_issues.md`.
  2. Consequence: `Zs` can never exceed 0.586·α·ws from stored water, and the negative-discriminant branch (`return ws*a`) is unreachable for storage capped at `sur_store(ws·α)`. In the calibrated 12167000 run, `Zs` reaches 0.34·α·ws, where the round-trip error is about 10%. Each step goes `ST → sur_z → Zs → sur_q`, so surface discharge is biased low relative to stored water.
  3. The manual's eq. 4 is not this relation either (§2.1), so the manual gives no way to tell which side is "right".

### 2.3 `sur_q(lb, a, ks, z)`: surface discharge

- **Manual:** eq. 9, p. 8 (also table 1): **Qs(t) = 2 Lb Ks α Zs(t)**. Darcy's law at the section where the water surface meets the land surface, with gradient α (p. 5, p. 7 "Storage-Discharge Relations"), on both sides of the channel.
- **Code:** `utilities.py:92–118`, `R:83`: `2·lb·z·a·ks`. **Matches eq. 9 exactly.**
- **Ambiguities**
  1. Table 2 says sur_q uses "the parameters for **equation 11**". Eq. 11 is the surface water balance; it should say eq. 9. This is a cross-reference error.
  2. The argument is named `z`. R's comment calls it "water surface elevation at surface of surface zone (datum is channel)". Under the manual's definition this is Zs at the Xs section. Same quantity, different words; adopt the manual's.
  3. Eq. 20 (p. 12) gives a different surface discharge inside the impulse algorithm: **Qs(t) = Xb [Zs(t−1) + F(t)/POR] Ks α**. It has no factor 2, uses Xb instead of Lb, and treats F/POR as a rise in water level, although F is a volume rate (table 1, eq. 14), so it lacks a division by area. The code does not implement eq. 20. Its impulse loop uses `ss → sur_z → sur_q` (`bfs.py:131–134`). Eq. 20 is dimensionally inconsistent as printed.

### 2.4 `dir_q(lb, a, z, i)`: direct runoff from saturated land surface

- **Manual:** eq. 19, p. 12: **Qd(t) = 2 Lb (Zs(t)/α) I(t)**. Eq. 15, p. 8 (table 1): **Qd′(t*) = 2 I(t*) Wb Xs(t*)**, with `Xs = Zs/α` (eq. 3). Inverse used to size the impulse: eq. 13, **I(t*) = 2 ε(t*)/(Wb Xs(t*))**.
- **Code:** `utilities.py:121–148`, `R:86`: `2·lb·z/a·i`. **Matches eq. 19 exactly.**
- **Ambiguities and departures**
  1. Eqs. 15 and 19 disagree: Wb·Xs vs. Lb·Xs. In the manual's geometry the saturated strip is Xs wide on each side and Lb long, so eq. 19 (and the code) is the consistent one. Eq. 15 is flagged.
  2. Eq. 13 is not the inverse of eq. 15: `Qd′ = 2 I Wb Xs` gives `I = ε/(2 Wb Xs)`, not `2ε/(Wb Xs)`. That is a factor-of-4 inconsistency inside the manual.
  3. The code's *initial* impulse guess `I = etaest/(2·lb·ws)` (`bfs.py:120`) spreads ε over the whole surface-zone area (2·Lb·Wb/2) instead of the saturated strip of eq. 13. It is only the starting point of the geometric search, but it departs from eq. 13 either way.
  4. Total direct runoff in `bfs` (`bfs.py:135–137, 204–206`) is
     `dir_q(zs_in, I) + dir_q(Zs − zs_in, I/2) + max(2·Lb·(ws − zs_in/α)·(I − Ks), 0)`.
     - The third term is the manual's "portion of the impulse that cannot be stored, 2 Xb (Wb/2 − Zs/α) I(t) − F(t)" (p. 12), with Lb in place of Xb and with F evaluated at Ks (see §2.5).
     - The second term (saturated-area expansion × **half** the impulse) matches the manual's statement that direct runoff "is updated at the end of the time step to account for the increase in the area of surface saturation" (p. 12). **The factor ½ is not in the manual.**

### 2.5 `infiltration(lb, ws, ks, a, zs, i)`: infiltration through unsaturated land surface

- **Manual:** eq. 14, p. 8 (table 1): **F(t) = 2 Xb(t) [Wb/2 − Zs(t)/α] min[Kz, I(t)]**. Prose p. 12: "the product of the unsaturated surface area and the smaller of **Kz** or the impulse … limited to storage available in the surface reservoir at the beginning of the time step". Table 2: `infiltration(Lb, Wb/2, **Ks**, α, Zs, I)`. Table 5: Ks is used for "Infiltration into and discharge from surface reservoir"; Kz for "Recharge … into base reservoir".
- **Code:** `utilities.py:151–183`, `R:89`: `2·lb·(ws − zs/a)·min(i, ks)`.
- **Departures and ambiguities**
  1. **Rate cap Ks vs. Kz:** eq. 14 and the p. 12 prose say Kz; tables 2 and 5 say Ks; the code (and R) uses Ks. **The manual contradicts itself.** The code follows the tables.
  2. **Length factor Xb vs. Lb:** eq. 14 (and the p. 12 "cannot be stored" expression) use Xb; the code uses Lb. This is the same issue as eq. 4. The unsaturated land surface in the manual's geometry runs the full channel length, so Lb is the consistent reading, but the manual writes Xb three times (eqs. 4, 14, and p. 12).
  3. The area factor `2·(Wb/2 − Zs/α)` = 2·(Wb/2 − Xs) matches the manual's "unsaturated portion of the land surface" exactly.
  4. **Storage limit:** p. 12 limits infiltration to available storage at the start of the step. The code does this in the caller (`min(…, ssa)`, `bfs.py:131,140,151`). P. 11 says instead that "infiltration and recharge are limited to the sum of available storage **and discharge from** the receiving reservoir". The code adds discharge (`+ qb`) for recharge only, not for infiltration.
  5. In the impulse search the code evaluates infiltration at the mean `(zs_in + zs)/2` (`bfs.py:131`). The manual doesn't specify which Zs to use.

### 2.6 `recharge(lb, xb, ws, kz, zs, por)`: surface → base reservoir

- **Manual:** eq. 16, p. 9 (body text): **R(t) = [Lb − Xb(t)] Wb min[Kz, n Zs(t)]**. Table 1 writes the same with **POR** in place of `n`. Prose: "the product of the unsaturated area of the base reservoir, (Lb – Xb(t)) Wb, and the smaller of either the vertical hydraulic conductivity, Kz, or the depth of water in the surface reservoir, Zs".
- **Code:** `utilities.py:186–218`, `R:92`: `(lb − xb)·2·ws·min(zs·por, kz)` = `(Lb − Xb)·Wb·min(Kz, POR·Zs)`. **Matches eq. 16 / table 1 exactly** (2·ws = Wb).
- **Ambiguities**
  1. `n` in eq. 16 is not defined anywhere; table 1 shows it means POR.
  2. The prose says the cap is the depth **Zs**; the equation says **POR·Zs** (drainable water depth). The code uses POR·Zs.
  3. `min(Kz, POR·Zs)` compares a rate [L/T] with a depth [L]. This is consistent only per unit time step, so the manual carries the same implicit Δt = 1 as the code (issue M8 is therefore in the manual, not a code departure).
  4. The storage limit (`min(…, sba + qb_in)`, `bfs.py:110,153`) matches p. 12 ("limited to the storage available in the base reservoir and the initial estimate of baseflow").

---

## 3. Summary

| Kernel | Manual eq. | Code = manual? | Issue |
|---|---|---|---|
| `sur_store` | 4 (p. 5) | **No** | Manual has Xb and a different bracket. Eq. 4 is inconsistent with the manual's own geometry (ambiguous); the code is consistent with eqs. 9/16/19 |
| `sur_z` | table 2 → eq. 4; p. 5 → eq. 3 | **No** | **Not the inverse of `sur_store` (R bug, verified; up to 41% error)**; the manual is unclear which equation it implements |
| `sur_q` | 9 (p. 8) | Yes | Table 2 cites eq. 11 (typo); eq. 20 is an inconsistent alternative that the code doesn't use |
| `dir_q` | 19 (p. 12); 15 (p. 8) | Yes (eq. 19) | Eq. 15 uses Wb (conflicts with 19); eq. 13 is not the inverse of 15; the ½ factor on the expansion term is undocumented |
| `infiltration` | 14 (p. 8) | **No** | Manual: Xb and Kz; code: Lb and Ks. The manual contradicts itself (tables 2/5 agree with the code) |
| `recharge` | 16 (p. 9) | Yes | `n` undefined in eq. 16 (= POR per table 1); prose says Zs, equation says POR·Zs; rate-vs-depth mix |

Cross-cutting manual ambiguities:
- **Lb's role:** table 4 says "width of base reservoir", eq. 9 says "width of surface reservoir", and p. 4 says the base reservoir has "the same length and width as the basin (Lb and Wb)". Table 4 is most likely an error for "surface reservoir".
- **Xb in surface-reservoir expressions** (eqs. 4, 14, 20 and the p. 12 runoff expression) conflicts with Lb in eqs. 9, 16, 19 and table 2. The code uses Lb throughout.
- **Initial conditions:** p. 12 says Qb(1) = Qthresh, and storage is found "using (2) and (3)" and "(4) and (5)". The code uses `min(Q(1), Qmean)` (`bfs.py:42,401`; R does the same), and the manual has no eq. 2. This is outside the kernels but logged as M12.

Nothing in the code has changed. The proposed docstrings in Phase 2 will cite the manual equation numbers above, use the manual's definition of Zs, and carry a "departs from manual" note where this table says No.
