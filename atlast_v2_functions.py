# AT-LAST v2.0.0 — auto-extracted functions
import os, json, warnings, numpy as np, pandas as pd
from datetime import datetime, timezone
from scipy.signal import argrelextrema
warnings.filterwarnings('ignore')
try:
    from gatspy.periodic import LombScargleMultiband
except ImportError:
    import subprocess
    subprocess.run(['pip','install','gatspy','--break-system-packages','-q'])
    from gatspy.periodic import LombScargleMultiband

def get_object_data(df, provid):
    """Extract and clean photometry for a single object."""
    obj = df[df["provid"] == provid].copy()
    band_map = {"g": "Lg", "r": "Lr", "i": "Li", "u": "Lu"}
    obj["band"] = obj["band"].replace(band_map)
    obj = obj[obj["rmsmag"] <= CFG["max_rmsmag"]]
    obj = obj.sort_values("t_hr").reset_index(drop=True)
    return obj


def normalise_multiband(obj, ref_band="Lr"):
    """Subtract per-band median offset to bring all bands to same zero point."""
    if ref_band not in obj["band"].unique():
        ref_band = obj["band"].value_counts().index[0]
    ref_median = obj[obj["band"] == ref_band]["mag"].median()
    obj = obj.copy()
    for band in obj["band"].unique():
        offset = obj[obj["band"] == band]["mag"].median() - ref_median
        obj.loc[obj["band"] == band, "mag"] -= offset
    return obj, ref_band


def get_score_threshold(n_obs):
    """
    Dynamic LS score threshold based on observation count.
    More observations → lower threshold needed (signal more reliable).
    """
    if n_obs >= 200:
        return 0.25
    elif n_obs >= 100:
        return 0.35
    elif n_obs >= 50:
        return 0.45
    else:
        return 0.50


def run_lomb_scargle_multiband(obj_df, cfg):
    """
    True multiband Lomb-Scargle (VanderPlas & Ivezic 2015) via gatspy.
    Fits a shared period across all bands simultaneously.

    Note: gatspy returns a score (0-1, R²-like), NOT a FAP.
    Higher score = better fit. Threshold is empirically set via
    get_score_threshold() based on observation count.

    Returns: freq, scores, periods, top_periods, top_scores, score_max
    """
    t     = obj_df["t_hr"].values
    mag   = obj_df["mag"].values
    err   = obj_df["rmsmag"].values
    bands = obj_df["band"].values

    t_span     = t.max() - t.min()
    period_max = min(cfg["period_max_hr"], t_span / 2.0)
    period_min = cfg["period_min_hr"]

    if period_max <= period_min or t_span < period_min:
        return None, None, None, np.array([]), np.array([]), 0.0

    f_min  = 1.0 / period_max
    f_max  = 1.0 / period_min
    n_freq = int(10 * f_max / f_min)
    n_freq = max(n_freq, 1000)
    n_freq = min(n_freq, 50000)

    frequencies = np.linspace(f_min, f_max, n_freq)
    periods     = 1.0 / frequencies

    model  = LombScargleMultiband(Nterms_base=1, Nterms_band=1)
    model.fit(t, mag, err, bands)
    scores = model.score(periods)

    peak_idx = argrelextrema(scores, np.greater, order=5)[0]
    if len(peak_idx) == 0:
        peak_idx = np.array([np.argmax(scores)])
    peak_idx    = peak_idx[np.argsort(scores[peak_idx])[::-1]]
    top_idx     = peak_idx[:cfg["n_top_periods"]]
    top_periods = periods[top_idx]
    top_scores  = scores[top_idx]
    score_max   = float(top_scores[0]) if len(top_scores) > 0 else 0.0

    return frequencies, scores, periods, top_periods, top_scores, score_max


def fit_fourier(t_hr, mag, rmsmag, period_hr, max_order=5):
    """
    Fit Fourier model of orders 1..max_order, select best by BIC.
    Returns dict with order, coeffs, bic, chi2, residuals, phase.
    """
    phase    = (t_hr % period_hr) / period_hr
    best_bic = np.inf
    best     = {}

    for order in range(1, max_order + 1):
        cols = [np.ones(len(phase))]
        for k in range(1, order + 1):
            cols.append(np.sin(2 * np.pi * k * phase))
            cols.append(np.cos(2 * np.pi * k * phase))
        A = np.column_stack(cols)
        w = 1.0 / rmsmag**2
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                A * w[:, None], mag * w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals = mag - A @ coeffs
        chi2      = np.sum((residuals / rmsmag)**2)
        bic       = chi2 + (2 * order + 1) * np.log(len(mag))
        if bic < best_bic:
            best_bic = bic
            best = {"order": order, "coeffs": coeffs, "bic": bic,
                    "chi2": chi2, "residuals": residuals, "phase": phase}
    return best


def count_extrema(t_hr, mag, rmsmag, period_hr, order):
    """Count peaks + troughs in ONE phase-folded cycle of Fourier model."""
    phase_fine = np.linspace(0, 1, 1000)
    cols_fine  = [np.ones(len(phase_fine))]
    for k in range(1, order + 1):
        cols_fine.append(np.sin(2 * np.pi * k * phase_fine))
        cols_fine.append(np.cos(2 * np.pi * k * phase_fine))
    A_fine = np.column_stack(cols_fine)

    phase      = (t_hr % period_hr) / period_hr
    cols_data  = [np.ones(len(phase))]
    for k in range(1, order + 1):
        cols_data.append(np.sin(2 * np.pi * k * phase))
        cols_data.append(np.cos(2 * np.pi * k * phase))
    A_data = np.column_stack(cols_data)
    w      = 1.0 / rmsmag**2
    coeffs, _, _, _ = np.linalg.lstsq(
        A_data * w[:, None], mag * w, rcond=None)

    model        = A_fine @ coeffs
    sign_changes = np.sum(np.diff(np.sign(np.diff(model))) != 0)
    return sign_changes


def expected_extrema(fourier_order):
    """Expected extrema per cycle = 2 * fourier_order."""
    return 2 * fourier_order


def resolve_period(t, mag, err, best_period, top_periods,
                   fourier, obs_extrema, exp_extrema, cfg):
    """
    Alias resolution — BIC primary, extrema secondary.

    If base period passes extrema → keep it.
    If base period fails extrema → find best alternative by BIC.
    Accept any candidate with better BIC when base already fails.
    """
    base_extrema_ok = (
        abs(obs_extrema - exp_extrema) <= cfg["extrema_tolerance"])

    if base_extrema_ok:
        return best_period, fourier, "none", True

    best_bic     = fourier["bic"]
    period_final = best_period
    best_fourier = fourier
    alias_tested = "none"
    extrema_ok   = False

    candidates = []
    for ratio in cfg["alias_ratios"]:
        alias_p = best_period * ratio
        if cfg["period_min_hr"] < alias_p < cfg["period_max_hr"]:
            candidates.append((alias_p, f"x{ratio:.2f}"))
    for p in top_periods[1:]:
        candidates.append((p, "alt_peak"))

    for cand_p, label in candidates:
        f = fit_fourier(t, mag, err, cand_p, cfg["max_fourier_order"])
        if not f:
            continue
        e     = count_extrema(t, mag, err, cand_p, f["order"])
        e_exp = expected_extrema(f["order"])
        e_ok  = abs(e - e_exp) <= cfg["extrema_tolerance"]

        # Accept if BIC improves (base already failing, so any improvement ok)
        if f["bic"] < best_bic and (e_ok or not base_extrema_ok):
            best_bic     = f["bic"]
            period_final = cand_p
            best_fourier = f
            alias_tested = label
            extrema_ok   = e_ok

    return period_final, best_fourier, alias_tested, extrema_ok


def bootstrap_period(t_hr, mag, rmsmag, period_hr, cfg):
    """
    Bootstrap stability: fraction of resamples recovering same period
    or a recognised alias (P/2, 2P, P/3, 3P).
    Returns None if n_bootstrap=0 (disabled).
    """
    n_boot = cfg.get("n_bootstrap", 0)
    if n_boot == 0:
        return None

    hits = 0
    n    = len(t_hr)
    equivalent = [
        period_hr,
        period_hr * 2.0,
        period_hr * 0.5,
        period_hr * 3.0,
        period_hr / 3.0,
    ]

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            obj_b = pd.DataFrame({
                "t_hr"   : t_hr[idx],
                "mag"    : mag[idx],
                "rmsmag" : rmsmag[idx],
                "band"   : np.full(n, "Lr"),
            })
            _, _, _, top_b, top_s, _ = run_lomb_scargle_multiband(obj_b, cfg)
            if len(top_b) == 0:
                continue
            for found_p in top_b[:3]:
                for eq_p in equivalent:
                    if eq_p > 0 and 0.9 < found_p / eq_p < 1.1:
                        hits += 1
                        break
                else:
                    continue
                break
        except Exception:
            pass

    return hits / n_boot


def process_object(df, provid, cfg):
    """
    Full pipeline for a single object.

    Steps:
    1. Load and clean photometry
    2. Multiband normalisation
    3. True multiband Lomb-Scargle (gatspy)
    4. Dynamic score threshold
    5. Fourier model fitting + BIC selection
    6. Extrema gate
    7. Alias resolution (BIC-primary, extrema-soft)
    8. Bootstrap stability (optional)
    9. Final flag assignment

    Returns dict with period, flag, diagnostics.
    """
    result = {
        "provid"          : provid,
        "status"          : "FAIL",
        "period_hr"       : None,
        "period_flag"     : None,
        "fourier_order"   : None,
        "bic"             : None,
        "ls_score"        : None,
        "boot_fraction"   : None,
        "n_obs_used"      : None,
        "n_bands"         : None,
        "mag_range"       : None,
        "extrema_observed": None,
        "extrema_expected": None,
        "extrema_pass"    : None,
        "alias_tested"    : None,
    }

    try:
        # Step 1: Load and clean
        obj = get_object_data(df, provid)
        if len(obj) < cfg["min_obs"]:
            result["status"] = "INSUFFICIENT_OBS"
            return result

        # Step 2: Multiband normalisation
        obj, ref_band = normalise_multiband(obj, cfg["ref_band"])
        n_obs = len(obj)
        result["n_obs_used"] = n_obs
        result["n_bands"]    = obj["band"].nunique()
        result["mag_range"]  = float(np.ptp(obj["mag"].values))

        # Step 3: Multiband Lomb-Scargle
        freq, scores, periods, top_periods, top_scores, score_max = \
            run_lomb_scargle_multiband(obj, cfg)

        if freq is None or len(top_periods) == 0:
            result["status"] = "NO_PERIOD_DETECTED"
            return result

        # Step 4: Dynamic score threshold
        threshold         = get_score_threshold(n_obs)
        result["ls_score"] = float(score_max)

        if score_max < threshold:
            result["status"] = "NO_PERIOD_DETECTED"
            return result

        t   = obj["t_hr"].values
        mag = obj["mag"].values
        err = obj["rmsmag"].values

        # Step 5: Fourier model fitting
        best_period = top_periods[0]
        fourier     = fit_fourier(t, mag, err, best_period,
                                  cfg["max_fourier_order"])
        if not fourier:
            result["status"] = "FOURIER_FIT_FAILED"
            return result

        # Step 6: Extrema gate
        obs_extrema = count_extrema(t, mag, err, best_period,
                                    fourier["order"])
        exp_extrema = expected_extrema(fourier["order"])

        # Step 7: Alias resolution
        period_final, fourier_final, alias_tested, extrema_ok = \
            resolve_period(t, mag, err, best_period, top_periods,
                           fourier, obs_extrema, exp_extrema, cfg)

        obs_ext_final = count_extrema(t, mag, err, period_final,
                                      fourier_final["order"])
        exp_ext_final = expected_extrema(fourier_final["order"])

        # Step 8: Bootstrap stability
        boot_frac = bootstrap_period(t, mag, err, period_final, cfg)

        # Step 9: Flag assignment
        if boot_frac is None:
            flag = "UNVERIFIED"
        elif boot_frac >= cfg["boot_min_fraction"]:
            flag = "RELIABLE"
        elif boot_frac >= 0.3:
            flag = "TENTATIVE"
        else:
            flag = "UNRELIABLE"

        result.update({
            "status"          : "OK",
            "period_hr"       : float(period_final),
            "period_flag"     : flag,
            "fourier_order"   : fourier_final["order"],
            "bic"             : float(fourier_final["bic"]),
            "ls_score"        : float(score_max),
            "boot_fraction"   : boot_frac,
            "alias_tested"    : alias_tested,
            "extrema_observed": obs_ext_final,
            "extrema_expected": exp_ext_final,
            "extrema_pass"    : extrema_ok,
        })

    except Exception as e:
        result["status"] = f"ERROR: {str(e)[:80]}"

    return result


def load_photometry(path=None):
    """Load and prepare the full photometry dataset."""
    path = path or FILES["input_photometry"]
    df = pd.read_csv(path, parse_dates=["obstime"])
    df["obstime"] = pd.to_datetime(df["obstime"], utc=True)
    t0 = df["obstime"].min()
    df["t_hr"]  = (df["obstime"] - t0).dt.total_seconds() / 3600.0
    df["provid"] = df["provid"].str.strip()
    n_raw = len(df)
    df = df.dropna(subset=["mag", "band", "rmsmag"])
    print(f"Loaded {len(df):,} obs ({n_raw-len(df):,} dropped) "
          f"for {df['provid'].nunique():,} objects")
    print(f"Bands: {sorted(df['band'].unique())}")
    print(f"Date range: {df['obstime'].min().date()} → "
          f"{df['obstime'].max().date()}")
    return df


def build_batch_list(df_all, min_obs=10):
    """Build list of objects to process, sorted by priority."""
    stats = df_all.groupby("provid").agg(
        n_obs      = ("mag",    "count"),
        n_bands    = ("band",   "nunique"),
        bands      = ("band",   lambda x: ",".join(sorted(x.unique()))),
        mag_range  = ("mag",    lambda x: x.max() - x.min()),
        rmsmag_med = ("rmsmag", "median"),
        t_span_hr  = ("t_hr",   lambda x: x.max() - x.min()),
    ).reset_index()

    batch = stats[stats["n_obs"] >= min_obs].copy()
    batch["priority"] = (
        batch["n_obs"].clip(upper=50) / 50.0 * 0.5 +
        batch["mag_range"].clip(upper=1.0) * 0.5
    )
    batch = batch.sort_values("priority", ascending=False).reset_index(drop=True)
    print(f"Batch list: {len(batch):,} objects with ≥{min_obs} obs")
    return batch


def run_batch(df_all, provid_list, cfg, results_path, checkpoint_path,
              checkpoint_every=500):
    """
    Run pipeline on a list of objects with checkpointing.
    Resumes from checkpoint if one exists.
    """
    import time

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        df_done     = pd.read_csv(checkpoint_path)
        done_provids = set(df_done["provid"].tolist())
        results      = df_done.to_dict("records")
        print(f"Checkpoint: {len(done_provids):,} done, "
              f"{len(provid_list)-len(done_provids):,} remaining")
    else:
        done_provids = set()
        results      = []

    todo = [p for p in provid_list if p not in done_provids]
    print(f"Processing {len(todo):,} objects...")

    t_start = time.time()

    for i, provid in enumerate(todo):
        r = process_object(df_all, provid, cfg)
        match, matched_p = is_match(provid, r["period_hr"])

        results.append({
            "provid"         : provid,
            "period_hr"      : r["period_hr"],
            "period_flag"    : r["period_flag"],
            "ls_score"       : r["ls_score"],
            "fourier_order"  : r["fourier_order"],
            "bic"            : r["bic"],
            "status"         : r["status"],
            "n_obs"          : r["n_obs_used"],
            "n_bands"        : r["n_bands"],
            "mag_range"      : r["mag_range"],
            "extrema_pass"   : r["extrema_pass"],
            "alias_tested"   : r["alias_tested"],
            "validated_match": match,
            "matched_period" : matched_p,
        })

        if (i + 1) % checkpoint_every == 0:
            df_check  = pd.DataFrame(results)
            df_check.to_csv(checkpoint_path, index=False)
            elapsed   = time.time() - t_start
            rate      = (i + 1) / elapsed
            eta_hr    = (len(todo) - i - 1) / rate / 3600
            pct_ok    = (df_check["status"] == "OK").mean() * 100
            print(f"  [{i+1:>6,}/{len(todo):,}]  "
                  f"elapsed={elapsed/3600:.2f}hr  "
                  f"eta={eta_hr:.1f}hr  "
                  f"ok={pct_ok:.1f}%")

    df_results = pd.DataFrame(results)
    df_results.to_csv(results_path, index=False)
    df_results.to_csv(checkpoint_path, index=False)

    total_time = time.time() - t_start
    ok     = df_results[df_results["status"] == "OK"]
    no_det = df_results[df_results["status"] == "NO_PERIOD_DETECTED"]

    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE")
    print(f"{'='*50}")
    print(f"Total objects      : {len(df_results):,}")
    print(f"Period found       : {len(ok):,}  ({100*len(ok)/len(df_results):.1f}%)")
    print(f"No period detected : {len(no_det):,}  ({100*len(no_det)/len(df_results):.1f}%)")
    print(f"Total time         : {total_time/3600:.2f} hr")
    print(f"Avg time/object    : {total_time/len(df_results):.1f}s")
    print(f"Results → {results_path}")

    return df_results


def is_match(provid, pipeline_period, tolerance=0.10):
    """
    Check if pipeline period matches any valid period for this object.
    Also checks P/2, 2P, P/3, 3P of each valid period.
    Returns (match: bool, matched_period: float or None)
    """
    if pipeline_period is None:
        return False, None

    valid = VALID_PERIODS.get(provid, [])
    if not valid:
        return False, None

    expanded = []
    for p in valid:
        expanded.extend([p, p*2.0, p*0.5, p*3.0, p/3.0])

    for vp in expanded:
        if vp > 0 and abs(pipeline_period - vp) / vp <= tolerance:
            return True, vp

    return False, None


VALID_PERIODS = {'2025 MA19': [8.867, 10.913, 14.201], '2025 MA45': [1.633, 0.817], '2025 MA46': [5.867, 5.225, 6.685], '2025 MC34': [8.376, 5.075, 9.188, 10.144, 15.216], '2025 MD38': [9.489, 7.906, 11.856, 23.696, 47.392], '2025 MD40': [4.371, 4.006, 4.811], '2025 MD67': [3.331, 3.311, 5.879], '2025 ME15': [8.072, 4.849, 6.041, 6.902], '2025 ME24': [4.152], '2025 ME68': [0.9], '2025 MF76': [2.028, 1.865, 1.869, 2.217], '2025 MG17': [4.278], '2025 MG56': [0.3], '2025 MH40': [6.878, 8.023, 12.028, 48.2], '2025 MH69': [7.82], '2025 MH75': [4.231], '2025 MH86': [4.391, 4.019, 4.835], '2025 MJ13': [3.097], '2025 MJ21': [3.942], '2025 MJ23': [5.182], '2025 MJ30': [5.594, 5.009], '2025 MJ71': [0.031], '2025 MJ79': [0.423], '2025 MK23': [10.639, 3.09, 6.18, 9.273, 14.191, 17.732, 20.793], '2025 MK41': [0.063], '2025 MK68': [5.017, 4.548, 5.576, 5.604, 6.266, 8.357], '2025 MK83': [6.087, 5.391, 5.398], '2025 MK88': [2.523], '2025 ML10': [7.037, 4.918, 6.151, 8.196], '2025 ML17': [6.723, 5.253, 5.896, 7.827, 7.841], '2025 ML35': [21.289, 14.711, 21.662, 38.255, 38.424], '2025 ML52': [9.238, 11.439, 13.842, 17.159], '2025 ML53': [6.998, 6.104], '2025 MM37': [1.852, 1.72, 1.724, 2.006], '2025 MM81': [1.1], '2025 MM82': [5.021, 5.609], '2025 MN25': [0.4], '2025 MN37': [6.825, 9.034], '2025 MN45': [0.031], '2025 MN7': [10.6], '2025 MO35': [6.281, 5.551], '2025 MO39': [6.171, 4.952, 7.056, 7.067, 8.229], '2025 MO47': [9.132], '2025 MO79': [2.739, 3.094, 3.555, 4.912, 4.915, 5.481, 6.184], '2025 MP21': [5.502, 6.22], '2025 MP47': [2.207, 4.036, 4.409, 4.858, 4.861, 4.868], '2025 MP61': [5.569], '2025 MP67': [6.194, 3.54, 5.499, 5.52], '2025 MP71': [9.1], '2025 MQ58': [3.081], '2025 MR33': [3.494], '2025 MS34': [2.315, 2.209, 2.432], '2025 MS7': [4.56, 3.832, 4.165, 5.042], '2025 MT24': [8.894, 6.673], '2025 MU10': [6.5], '2025 MU15': [0.4], '2025 MU24': [2.216], '2025 MU59': [8.189, 4.093], '2025 MU8': [0.823, 0.838], '2025 MU9': [3.492], '2025 MV19': [7.4], '2025 MV31': [5.801, 5.168, 5.177, 5.19, 5.832, 6.497], '2025 MV38': [6.006, 4.797, 5.332, 6.858, 6.879], '2025 MV4': [5.266, 4.742, 5.917, 5.925, 7.9], '2025 MV46': [3.405], '2025 MV71': [0.2], '2025 MW70': [3.885, 3.336, 3.591, 4.205, 4.633, 4.643], '2025 MX34': [4.677], '2025 MX44': [1.151, 1.098], '2025 MX50': [1.94, 3.882], '2025 MX63': [8.287], '2025 MX69': [7.662, 5.769, 7.603, 9.109], '2025 MY23': [3.058], '2025 MY77': [7.6], '2025 MZ78': [1.2]}

CFG = {'atlast_version': '2.0.0', 'run_id': '20260318_214641', 'run_name': 'atlast_20260318_214641', 'min_obs': 6, 'min_obs_preferred': 10, 'max_rmsmag': 0.3, 'ref_band': 'Lr', 'period_min_hr': 0.01, 'period_max_hr': 100.0, 'ls_fap_threshold': 0.5, 'n_top_periods': 5, 'max_fourier_order': 5, 'alias_ratios': [2.0, 0.5, 3.0, 0.3333333333333333], 'extrema_tolerance': 1, 'n_bootstrap': 0, 'boot_min_fraction': 0.5, 'batch_min_obs': 10, 'batch_checkpoint_every': 500}

CFG_BATCH    = {**CFG, 'period_min_hr':0.01, 'period_max_hr':100.0, 'n_bootstrap':0}
CFG_STANDARD = {**CFG, 'period_min_hr':2.2,  'period_max_hr':100.0, 'n_bootstrap':0}
CFG_SUPERFAST= {**CFG, 'period_min_hr':0.01, 'period_max_hr':2.2,   'n_bootstrap':0}
print('AT-LAST v2.0.0 loaded.')
