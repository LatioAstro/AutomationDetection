import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import LC


ROOT = Path(__file__).resolve().parent.parent
PYTHON_FILES = ROOT / 'PythonFiles'
COSICSV = ROOT / 'COSICSV'
DEFAULT_DB_PATH = PYTHON_FILES / 'new_db_Aug2025.csv'
DEFAULT_FACTOR_TABLE = COSICSV / 'COSI_factors_all_updated_only_logpar_flare_updated_60deg_offaxis_last.csv'
DEFAULT_OUTPUT_DIR = ROOT / 'RemadeNov2025'
DEFAULT_INCREMENTAL_CACHE = ROOT / 'DownloadedLC' / 'incremental' / 'quiescent_background_cache.csv'
CATALOG_PATH = PYTHON_FILES / '4fgl-dr3_LCR.csv'

MJDREFI = 51910
MJDREFF = 7.428703703703703e-4
SECS_IN_DAY = 86400
VAR_INDEX_THRESHOLD = 21.67

IGNORE_LIST = [
    '4FGL J0010.6+2043', '4FGL J0222.0-1616', '4FGL J0228.0-3026', '4FGL J0312.8+0134',
    '4FGL J0358.9+6004', '4FGL J0824.7+5552', '4FGL J1031.6+6019', '4FGL J1209.8+1810',
    '4FGL J1446.7+1719', '4FGL J1635.6+3628', '4FGL J1647.5+4950', '4FGL J1716.1+6836',
    '4FGL J1724.9+7654', '4FGL J1808.1-5013', '4FGL J2256.0-2740', '4FGL J1222.5+0414',
    '4FGL J0405.6-1308', '4FGL J0336.4+3224', '4FGL J1337.6-1257', '4FGL J1924.8-2914',
    '4FGL J0024.7+0349', '4FGL J0239.7+0415', '4FGL J0904.6+5200', '4FGL J1118.2-0415',
    '4FGL J1205.7-2635', '4FGL J1324.9+4748', '4FGL J1445.9-1626', '4FGL J1559.9+2319',
    '4FGL J1650.3-5045', '4FGL J0116.0-1136', '4FGL J0200.6-6637', '4FGL J0224.9+1843',
    '4FGL J0304.5+3349', '4FGL J0347.7-3616', '4FGL J0427.3-3900', '4FGL J0450.3-4419',
    '4FGL J0617.6-4028', '4FGL J0746.4+2546', '4FGL J0805.4+6147', '4FGL J0943.7+6137',
    '4FGL J1043.2+2408', '4FGL J1200.7+2008', '4FGL J1659.7-3131', '4FGL J1747.6-5308',
    '4FGL J1824.5+0107', '4FGL J1912.1-0803', '4FGL J2040.5-1705', '4FGL J2120.6-1254',
    '4FGL J2148.6+0652', '4FGL J2149.7+1917', '4FGL J2207.6+0053', '4FGL J2358.0-4601',
    '4FGL J0034.0-4116', '4FGL J0035.8+6131', '4FGL J0137.9+5814', '4FGL J0152.2+3714',
    '4FGL J0156.5+3914', '4FGL J0342.2+3858', '4FGL J0353.0+5654', '4FGL J0401.7+2112',
    '4FGL J0407.5+0741', '4FGL J0453.3+2843', '4FGL J0455.7-4617', '4FGL J0512.8+4041',
    '4FGL J0635.6-7518', '4FGL J0638.7+5658', '4FGL J0647.7-6058', '4FGL J0713.0+5738',
    '4FGL J0723.5+2900', '4FGL J0747.3-3310', '4FGL J0806.5+4503', '4FGL J0836.5-2026',
    '4FGL J0911.0-5047', '4FGL J0923.5+3852', '4FGL J1001.1+2911', '4FGL J1015.6+5553',
    '4FGL J1016.0+0512', '4FGL J1018.1+1905', '4FGL J1045.8-2928', '4FGL J1136.2+3407',
    '4FGL J1158.5+4824', '4FGL J1225.6-7313', '4FGL J1238.5-1201', '4FGL J1322.2+0842',
    '4FGL J1440.0-1530', '4FGL J1454.4-3744', '4FGL J1513.4-3231', '4FGL J1516.9+1934',
    '4FGL J1656.0+2047', '4FGL J1753.6-5014', '4FGL J1808.2+3500', '4FGL J1834.7-5858',
    '4FGL J1912.0+1612', '4FGL J1917.7-6930', '4FGL J1942.1+4011', '4FGL J2130.4-4241',
    '4FGL J2143.1-3929', '4FGL J2146.4-1528', '4FGL J2219.2-0342', '4FGL J2359.2-3134',
    '4FGL J0059.4-5654', '4FGL J0132.1-0956', '4FGL J0156.3-2420', '4FGL J0205.7+6449',
    '4FGL J0240.5+6113', '4FGL J0259.0+0552', '4FGL J0308.4+0407', '4FGL J0333.0-3044',
    '4FGL J0405.4-6929', '4FGL J0448.7-2116', '4FGL J0521.2+1637', '4FGL J0534.5+2200',
    '4FGL J0537.5+0959', '4FGL J0540.0-7552', '4FGL J0622.9+3326', '4FGL J0833.3-4342c',
    '4FGL J0850.0+5108', '4FGL J0931.9+6737', '4FGL J0948.9+0022', '4FGL J1023.7+0038',
    '4FGL J1045.1-5940', '4FGL J1047.2+6740', '4FGL J1106.7+3623', '4FGL J1228.0-4853',
    '4FGL J1230.8+1223', '4FGL J1231.1-1412', '4FGL J1401.7-3217', '4FGL J1418.4+3543',
    '4FGL J1505.0+0326', '4FGL J1514.8+4448', '4FGL J1535.9+3743', '4FGL J1543.6+0452',
    '4FGL J1626.5-4406', '4FGL J1644.9+2620', '4FGL J1732.7-5050', '4FGL J1753.9+2443',
    '4FGL J1839.4-0553', '4FGL J1935.2+2029', '4FGL J2007.9-4432', '4FGL J2021.5+4026',
    '4FGL J2234.7+0943', '4FGL J2237.6-5126', '4FGL J0522.9-3628', '4FGL J0433.0+0522',
    '4FGL J0319.8+4130', '4FGL J0324.8+3412', '4FGL J1829.5+4845', '4FGL J2055.0-5218',
    '4FGL J1459.0+7140', '4FGL J1632.8-1048', '4FGL J1148.5+2629',
]

BAD_TIMES = {
    '4FGL J1941.3-6210': [58309.00001157408, 58526.00001157408, 58680.00001157408, 59051.00001157408, 59422.00001157408, 59793.00001157408],
    '4FGL J0805.4+6147': [59576.00001157408, 59947.00001157408, 60472.00001157408],
    '4FGL J1153.4+4931': [58785.00001157408],
    '4FGL J0116.0-1136': [58239.00001157408],
    '4FGL J0102.8+5824': [60227.00001157408],
    '4FGL J0359.6+5057': [58638.00001157408],
    '4FGL J0533.3+4823': [58260.00001157408, 59002.00001157408, 59744.00001157408],
    '4FGL J0534.5+2201s': [59744.00001157408, 60115.00001157408],
    '4FGL J0622.9+3326': [60479.00001157408],
    '4FGL J1535.8-4730': [58792.00001157408, 59163.00001157408],
    '4FGL J1555.2-4149': [58421.00001157408, 59163.00001157408],
}


def compute_mdp99(src_counts, bkg_counts, average_mu=0.3):
    src_counts = np.asarray(src_counts, dtype=float)
    bkg_counts = np.asarray(bkg_counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mdp99 = (4.29 / (average_mu * src_counts)) * np.sqrt(src_counts + bkg_counts) * 100.0
    return mdp99


def load_weekly_database(database_path=DEFAULT_DB_PATH, cadence='weekly'):
    cadence = normalize_cadence(cadence)
    readed = pd.read_csv(database_path, sep=',', na_filter=True)
    readed = readed.fillna(-3333)
    cadence_df = readed.loc[readed['cadence'] == cadence].copy()
    invalid_cols = ['photon_flux2', 'photon_flux_error2']
    cadence_df.loc[cadence_df['ts2'] <= 4.0, ['photon_flux2', 'photon_flux_error2']] = -3333
    cadence_df.loc[cadence_df['photon_flux_error2'] > cadence_df['photon_flux2'], ['photon_flux2', 'photon_flux_error2']] = -3333
    cadence_df.loc[cadence_df['photon_flux2'] > 1e-4, ['photon_flux2', 'photon_flux_error2']] = -3333
    cadence_df.loc[cadence_df['photon_flux2'] < 1e-10, ['photon_flux2', 'photon_flux_error2']] = -3333
    cadence_df.loc[(cadence_df['ts2'] <= 25.0) & (cadence_df['photon_flux2'] > 1e-6), invalid_cols] = -3333
    return cadence_df.reset_index(drop=True)


def normalize_cadence(cadence):
    cadence_key = str(cadence).strip().lower()
    if cadence_key in {'3day', '3-day', 'daily'}:
        return 'daily'
    if cadence_key in {'weekly', 'monthly'}:
        return cadence_key
    raise ValueError(f'Unsupported cadence: {cadence}')


def load_source_class_sets(catalog_path=CATALOG_PATH):
    catalog = pd.read_csv(catalog_path)
    catalog = catalog.loc[catalog['Variability_Index'] >= VAR_INDEX_THRESHOLD].copy()
    catalog['CLASS1'] = catalog['CLASS1'].astype(str)
    return {
        'FSRQ': set(catalog.loc[catalog['CLASS1'].str.lower() == 'fsrq', 'Source_Name']),
        'BLL': set(catalog.loc[catalog['CLASS1'].str.lower() == 'bll', 'Source_Name']),
        'BCU': set(catalog.loc[catalog['CLASS1'].str.lower() == 'bcu', 'Source_Name']),
    }


def classify_source(source_name, class_sets):
    if source_name in class_sets['FSRQ']:
        return 'FSRQ'
    if source_name in class_sets['BLL']:
        return 'BLL'
    if source_name in class_sets['BCU']:
        return 'BCU'
    return 'None'


def cadence_step_days(cadence):
    cadence = normalize_cadence(cadence)
    if cadence == 'daily':
        return 3.0
    if cadence == 'weekly':
        return 7.0
    if cadence == 'monthly':
        return 30.0
    raise ValueError(f'Unsupported cadence: {cadence}')


def remove_bad_times(time_values, flux_values, error_values, source_name):
    badtimes = BAD_TIMES.get(source_name)
    if not badtimes:
        return time_values, flux_values, error_values
    mask = ~np.isin(time_values, np.asarray(badtimes, dtype=float))
    return time_values[mask], flux_values[mask], error_values[mask]


def build_source_arrays(cadence_df, source_name):
    sourcearray = cadence_df[cadence_df['source_name'] == source_name].reset_index(drop=True)
    sourcearray = sourcearray[sourcearray['photon_flux2'] != -3333].reset_index(drop=True)
    if sourcearray.empty:
        raise ValueError(f'No valid bins remain for {source_name}.')

    average_flux = float(np.nanmean(sourcearray['photon_flux2']))
    sourcearray = sourcearray[sourcearray['photon_flux2'] <= 100 * average_flux].reset_index(drop=True)
    if sourcearray.empty:
        raise ValueError(f'All bins were removed as outliers for {source_name}.')

    time = sourcearray['tmin'].to_numpy(dtype=float) / SECS_IN_DAY + MJDREFI
    photon_flux = sourcearray['photon_flux2'].to_numpy(dtype=float)
    errors = sourcearray['photon_flux_error2'].to_numpy(dtype=float)
    time, photon_flux, errors = remove_bad_times(time, photon_flux, errors, source_name)
    if len(time) == 0:
        raise ValueError(f'All bins were removed by bad-time masking for {source_name}.')

    return sourcearray, time, photon_flux, errors, average_flux


def initial_threshold(photon_flux, percent):
    maxflux = float(np.max(photon_flux))
    minflux = float(np.min(photon_flux))
    delta_flux = maxflux - minflux
    return minflux + delta_flux * percent


def build_lightcurve_from_arrays(time_values, flux_values, error_values, source_name):
    return LC.LightCurve(time_values, flux_values, error_values, name=source_name, time_format='mjd')


def quiescent_background_from_arrays(time_values, flux_values, error_values, source_name, percent, cadence='weekly'):
    if len(time_values) < 3:
        return np.nan, np.nan, np.nan
    sourcelightcurve = build_lightcurve_from_arrays(time_values, flux_values, error_values, source_name)
    thresholdflux = initial_threshold(flux_values, percent)
    sourcelightcurve.get_bblocks(gamma_value=0.05)
    sourcelightcurve.find_hop(method='baseline', lc_edges='add', baseline=thresholdflux)
    quiescent_background, qui_err = quiescent_background_finder(
        sourcelightcurve=sourcelightcurve,
        cadence=cadence,
        thresholdflux=thresholdflux,
        method='forward',
    )
    return quiescent_background, qui_err, thresholdflux


def load_quiescent_cache(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pd.DataFrame(
            columns=[
                'Name',
                'cadence',
                'percent',
                'quiescent_background',
                'quiescent_background_error',
                'detection_threshold',
                'points_used',
                'source_time_end_mjd',
                'updated_at_utc',
            ]
        )
    cache_df = pd.read_csv(cache_path)
    required_cols = {
        'Name',
        'cadence',
        'percent',
        'quiescent_background',
        'quiescent_background_error',
        'detection_threshold',
    }
    if not required_cols.issubset(set(cache_df.columns)):
        raise ValueError(f'Cache file {cache_path} is missing required columns: {sorted(required_cols)}')
    cache_df['Name'] = cache_df['Name'].astype(str)
    cache_df['cadence'] = cache_df['cadence'].astype(str)
    cache_df['percent'] = pd.to_numeric(cache_df['percent'], errors='coerce')
    cache_df['quiescent_background'] = pd.to_numeric(cache_df['quiescent_background'], errors='coerce')
    cache_df['quiescent_background_error'] = pd.to_numeric(cache_df['quiescent_background_error'], errors='coerce')
    cache_df['detection_threshold'] = pd.to_numeric(cache_df['detection_threshold'], errors='coerce')
    if 'lookback_weeks' not in cache_df.columns:
        cache_df['lookback_weeks'] = np.nan
    cache_df['lookback_weeks'] = pd.to_numeric(cache_df['lookback_weeks'], errors='coerce')
    return cache_df


def resolve_quiescent_background(source_name, time_values, flux_values, error_values, percent, cadence, cache_path, lookback_weeks):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = load_quiescent_cache(cache_path)
    cadence = normalize_cadence(cadence)
    lookback_weeks = float(lookback_weeks)

    existing = cache_df.loc[
        (cache_df['Name'] == source_name)
        & (cache_df['cadence'] == cadence)
        & np.isclose(cache_df['percent'], float(percent), atol=1e-12, equal_nan=False)
        & np.isclose(cache_df['lookback_weeks'], lookback_weeks, atol=1e-12, equal_nan=False)
    ]
    if not existing.empty:
        entry = existing.iloc[-1]
        if np.isfinite(entry['quiescent_background']):
            return {
                'quiescent_background': float(entry['quiescent_background']),
                'quiescent_background_error': float(entry['quiescent_background_error']) if np.isfinite(entry['quiescent_background_error']) else np.nan,
                'detection_threshold': float(entry['detection_threshold']) if np.isfinite(entry['detection_threshold']) else np.nan,
                'origin': 'cache',
                'cache_path': str(cache_path),
            }

    qb, qb_err, threshold = quiescent_background_from_arrays(
        time_values,
        flux_values,
        error_values,
        source_name=source_name,
        percent=percent,
        cadence=cadence,
    )

    new_entry = {
        'Name': source_name,
        'cadence': cadence,
        'percent': float(percent),
        'quiescent_background': float(qb) if np.isfinite(qb) else np.nan,
        'quiescent_background_error': float(qb_err) if np.isfinite(qb_err) else np.nan,
        'detection_threshold': float(threshold) if np.isfinite(threshold) else np.nan,
        'lookback_weeks': lookback_weeks,
        'points_used': int(len(time_values)),
        'source_time_end_mjd': float(np.max(time_values)) if len(time_values) else np.nan,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }

    if existing.empty:
        cache_df = pd.concat([cache_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        mask = (
            (cache_df['Name'] == source_name)
            & (cache_df['cadence'] == cadence)
            & np.isclose(cache_df['percent'], float(percent), atol=1e-12, equal_nan=False)
            & np.isclose(cache_df['lookback_weeks'], lookback_weeks, atol=1e-12, equal_nan=False)
        )
        cache_df = cache_df.loc[~mask].copy()
        cache_df = pd.concat([cache_df, pd.DataFrame([new_entry])], ignore_index=True)

    cache_df.to_csv(cache_path, index=False)
    return {
        'quiescent_background': float(new_entry['quiescent_background']) if np.isfinite(new_entry['quiescent_background']) else np.nan,
        'quiescent_background_error': float(new_entry['quiescent_background_error']) if np.isfinite(new_entry['quiescent_background_error']) else np.nan,
        'detection_threshold': float(new_entry['detection_threshold']) if np.isfinite(new_entry['detection_threshold']) else np.nan,
        'origin': 'computed',
        'cache_path': str(cache_path),
    }


def quiescent_background_finder(sourcelightcurve, cadence, thresholdflux, method='forward'):
    qui = copy.deepcopy(sourcelightcurve)

    if not hasattr(sourcelightcurve, 'hops'):
        sourcelightcurve.find_hop(method='baseline', lc_edges='add', baseline=thresholdflux)

    if sourcelightcurve.hops is None:
        return np.nan, np.nan

    mask = []
    for hop in sourcelightcurve.hops:
        start_idx = np.searchsorted(sourcelightcurve.time, hop.start_time)
        end_idx = np.searchsorted(sourcelightcurve.time, hop.end_time)
        if start_idx < end_idx:
            mask.extend(range(start_idx, end_idx))

    maskindices = np.array(mask, dtype=int)
    if maskindices.size > 0:
        qui.flux = np.delete(qui.flux, maskindices)
        qui.time = np.delete(qui.time, maskindices)
        qui.flux_error = np.delete(qui.flux_error, maskindices)

    if qui.flux.size == 0:
        return np.nan, np.nan

    baseaverage = []
    weights = []
    tempavg = []
    tdiff = cadence_step_days(cadence)

    for idx in range(len(qui.flux) - 1):
        tempavg.append(qui.flux[idx])
        contiguous = np.isclose(qui.time[idx + 1] - qui.time[idx], tdiff, atol=1e-6)
        if idx == len(qui.flux) - 2 or not contiguous:
            if tempavg:
                baseaverage.append(np.nanmean(tempavg))
                weights.append(len(tempavg))
            tempavg = []

    if baseaverage and weights:
        quiescent_background = float(np.average(baseaverage, weights=weights))
    else:
        quiescent_background = np.nan

    if len(qui.flux_error) > 0:
        qui_err = float(np.sqrt(np.sum((np.array(qui.flux_error) ** 2) * (1 / len(qui.flux_error)) ** 2)))
    else:
        qui_err = 0.0
    return quiescent_background, qui_err


def characterize_hops(hops, sourcelightcurve, source_type, cosi_bkg_rate, cosi_aeff, lat_aeff, ph_ratio, int_flux_ratio):
    rows = []
    for hop in hops:
        duration = float(hop.dur * SECS_IN_DAY)
        asymmetry = float(hop.asym)
        coverage = float(hop.dur / (sourcelightcurve.time[-1] - sourcelightcurve.time[0]))
        background_counts = float(duration * cosi_bkg_rate)
        cosi_ph_rate = float(np.mean(hop.flux) * lat_aeff * ph_ratio)
        fluence_cosi = float(np.mean(hop.flux) * int_flux_ratio * duration)
        avg_flux = float(np.mean(hop.flux) * int_flux_ratio)
        lc_avg_flux = float(np.mean(sourcelightcurve.flux) * int_flux_ratio)
        peak_flux = float(np.max(hop.flux) * int_flux_ratio)
        sourcecounts = float(cosi_ph_rate * duration / 3.22)
        rows.append(
            {
                'Name': hop.name,
                'Class': source_type,
                'Average_Photon_Flux_(ph/cm2/s)_(0.2-5_MeV)': avg_flux,
                'Photon_Fluence_(ph/cm2)_(0.2-5_MeV)': fluence_cosi,
                'Source_Counts(ph)_(0.2-5_MeV)': sourcecounts,
                'Background_Counts': background_counts,
                'Duration_(s)': duration,
                'Start_Time_(MJD)': float(hop.start_time),
                'Coverage': coverage,
                'MDP99_(%)': float(compute_mdp99(sourcecounts, background_counts)),
                'Asymmetry': asymmetry,
                'Average_Flux_of_Entire_Source_(0.2-5_MeV)': lc_avg_flux,
                'Peak_Flare_Flux_(0.2-5_MeV)': peak_flux,
                'Aeff_mean_COSI(cm2)': float(cosi_aeff),
                'Aeff_mean_LAT(cm2)': float(lat_aeff),
                'ph/s_ratio': float(ph_ratio),
                'Int_flux_ratio': float(int_flux_ratio),
            }
        )
    return rows


def analyze_source(cadence_df, factor_row, class_sets, percent, cosi_bkg_rate):
    source_name = str(factor_row['Name'])
    sourcearray, time, photon_flux, errors, average_flux = build_source_arrays(cadence_df, source_name)
    sourcelightcurve = LC.LightCurve(time, photon_flux, errors, name=source_name)
    thresholdflux = initial_threshold(photon_flux, percent)

    sourcelightcurve.get_bblocks(gamma_value=0.05)
    sourcelightcurve.find_hop(method='baseline', lc_edges='add', baseline=thresholdflux)
    quiescent_background, qui_err = quiescent_background_finder(
        sourcelightcurve=sourcelightcurve,
        cadence='weekly',
        thresholdflux=thresholdflux,
        method='forward',
    )

    summary = {
        'Name': source_name,
        'quiescent_background': quiescent_background,
        'quiescent_background_error': qui_err,
        'average_flux': float(average_flux),
        'average_flux_cosi_band': float(average_flux * float(factor_row['Int_flux_ratio'])),
        'thresholdflux': float(thresholdflux),
        'points_used': int(len(time)),
        'time_start_mjd': float(np.min(time)),
        'time_end_mjd': float(np.max(time)),
    }

    if pd.isna(quiescent_background):
        return summary, []

    sourcelightcurve = LC.LightCurve(time, photon_flux, errors, name=source_name)
    sourcelightcurve.get_bblocks(gamma_value=0.05)
    sourcelightcurve.find_hop(method='baseline', lc_edges='add', baseline=quiescent_background)
    hops = sourcelightcurve.hops or []
    source_type = classify_source(source_name, class_sets)
    flare_rows = characterize_hops(
        hops=hops,
        sourcelightcurve=sourcelightcurve,
        source_type=source_type,
        cosi_bkg_rate=cosi_bkg_rate,
        cosi_aeff=float(factor_row['Aeff_mean_COSI(cm2)']),
        lat_aeff=float(factor_row['Aeff_mean_LAT(cm2)']),
        ph_ratio=float(factor_row['ph/s_ratio']),
        int_flux_ratio=float(factor_row['Int_flux_ratio']),
    )
    summary['flare_count'] = len(flare_rows)
    return summary, flare_rows


def incremental_flare_scan(
    source_name,
    database_path=DEFAULT_DB_PATH,
    percent=0.3,
    cadence='weekly',
    lookback_weeks=30.0,
    detection_method='both',
    flare_threshold_multiplier=2.0,
    confirmed_sigma_threshold=2.0,
    consecutive_points=3,
    cache_path=DEFAULT_INCREMENTAL_CACHE,
):
    cadence = normalize_cadence(cadence)
    detection_method = str(detection_method).strip().lower()
    if detection_method not in {'original', 'sigma', 'both'}:
        raise ValueError(f'Unsupported detection method: {detection_method}')

    cadence_df = load_weekly_database(database_path, cadence=cadence)
    _, time_values, flux_values, error_values, average_flux = build_source_arrays(cadence_df, source_name)

    lookback_days = float(lookback_weeks) * 7.0
    latest_mjd = float(np.max(time_values))
    window_start_mjd = latest_mjd - lookback_days
    lookback_mask = time_values >= window_start_mjd
    time_values = time_values[lookback_mask]
    flux_values = flux_values[lookback_mask]
    error_values = error_values[lookback_mask]
    if len(time_values) < 2:
        raise ValueError(
            f'Not enough cadence bins in the last {lookback_weeks:g} weeks for {source_name} at cadence={cadence}.'
        )

    qb_result = resolve_quiescent_background(
        source_name=source_name,
        time_values=time_values,
        flux_values=flux_values,
        error_values=error_values,
        percent=percent,
        cadence=cadence,
        cache_path=cache_path,
        lookback_weeks=lookback_weeks,
    )
    source_qb = qb_result['quiescent_background']
    source_qb_err = qb_result['quiescent_background_error']
    source_detection_threshold = qb_result['detection_threshold']
    comparison_threshold = float(source_qb * flare_threshold_multiplier) if np.isfinite(source_qb) else np.nan
    qb_sigma = float(source_qb_err) if np.isfinite(source_qb_err) and source_qb_err > 0 else 0.0

    rows = []
    streak = 0
    streak_start_index = None
    confirmed_active = False
    confirmed_start_mjd = np.nan

    for current_index in range(1, len(time_values)):
        new_flux = float(flux_values[current_index])
        new_flux_error = float(error_values[current_index])
        previous_flux_value = float(flux_values[current_index - 1])
        above_double_qb = bool(np.isfinite(comparison_threshold) and new_flux > comparison_threshold)
        above_previous_flux = bool(new_flux > previous_flux_value)
        significance_sigma = np.sqrt((new_flux_error ** 2) + (qb_sigma ** 2))
        flux_sigma_delta = float((new_flux - source_qb) / significance_sigma) if np.isfinite(source_qb) and significance_sigma > 0 else np.nan
        confirmed_start_trigger = bool(
            np.isfinite(flux_sigma_delta)
            and flux_sigma_delta > float(confirmed_sigma_threshold)
            and new_flux > source_qb
        )

        potential_candidate = above_double_qb if detection_method in {'original', 'both'} else False

        if potential_candidate:
            streak += 1
            if streak_start_index is None:
                streak_start_index = current_index
        else:
            streak = 0
            streak_start_index = None

        flare_active = (streak >= consecutive_points) if detection_method in {'original', 'both'} else False
        if streak_start_index is None:
            flare_start_mjd = np.nan
        else:
            flare_start_mjd = float(time_values[streak_start_index])

        if confirmed_active:
            if np.isfinite(source_qb) and new_flux < source_qb:
                confirmed_active = False
                confirmed_start_mjd = np.nan
        elif confirmed_start_trigger and detection_method in {'sigma', 'both'}:
            confirmed_active = True
            confirmed_start_mjd = float(time_values[current_index])

        rows.append(
            {
                'Name': source_name,
                'previous_bin_count': int(current_index),
                'current_bin_count': int(current_index + 1),
                'new_point_mjd': float(time_values[current_index]),
                'new_point_flux': new_flux,
                'new_point_flux_error': new_flux_error,
                'previous_point_flux': previous_flux_value,
                'average_flux_full_series': float(average_flux),
                'previous_quiescent_background': float(source_qb) if np.isfinite(source_qb) else np.nan,
                'previous_quiescent_background_error': float(source_qb_err) if np.isfinite(source_qb_err) else np.nan,
                'current_quiescent_background': float(source_qb) if np.isfinite(source_qb) else np.nan,
                'current_quiescent_background_error': float(source_qb_err) if np.isfinite(source_qb_err) else np.nan,
                'previous_detection_threshold': float(source_detection_threshold) if np.isfinite(source_detection_threshold) else np.nan,
                'current_detection_threshold': float(source_detection_threshold) if np.isfinite(source_detection_threshold) else np.nan,
                'flare_flux_threshold': comparison_threshold,
                'above_previous_flux': above_previous_flux,
                'above_flare_flux_threshold': above_double_qb,
                'potential_flare_point': potential_candidate,
                'consecutive_potential_flare_points': int(streak),
                'flare_active': flare_active,
                'flare_start_mjd': flare_start_mjd,
                'flare_end_mjd': float(time_values[current_index]) if flare_active else np.nan,
                'confirmed_sigma_delta': flux_sigma_delta,
                'confirmed_sigma_threshold': float(confirmed_sigma_threshold),
                'confirmed_start_trigger': bool(confirmed_start_trigger and detection_method in {'sigma', 'both'}),
                'confirmed_flare_active': bool(confirmed_active),
                'confirmed_flare_start_mjd': float(confirmed_start_mjd) if np.isfinite(confirmed_start_mjd) else np.nan,
                'confirmed_flare_end_mjd': float(time_values[current_index]) if confirmed_active else np.nan,
                'cadence': cadence,
                'lookback_weeks': float(lookback_weeks),
                'detection_method': detection_method,
                'quiescent_background_origin': qb_result['origin'],
                'quiescent_background_cache_path': qb_result['cache_path'],
            }
        )

    return pd.DataFrame(rows)


def run_parameter_grid(
    database_path=DEFAULT_DB_PATH,
    factor_table_path=DEFAULT_FACTOR_TABLE,
    output_dir=DEFAULT_OUTPUT_DIR,
    percent_ranges=None,
    background_ranges=None,
    source_name=None,
):
    if percent_ranges is None:
        percent_ranges = [0.1, 0.3, 0.5]
    if background_ranges is None:
        background_ranges = [20, 10, 1]

    cadence_df = load_weekly_database(database_path)
    factor_table = pd.read_csv(factor_table_path, sep=',', na_filter=True)
    class_sets = load_source_class_sets()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_name:
        factor_table = factor_table.loc[factor_table['Name'] == source_name].reset_index(drop=True)
        if factor_table.empty:
            raise ValueError(f'{source_name} was not found in {factor_table_path}.')

    outputs = []
    for bkg in background_ranges:
        cosi_bkg_rate = float(bkg) * 0.25
        for percent in percent_ranges:
            flare_rows = []
            quiescent_rows = []
            for _, factor_row in factor_table.iterrows():
                source = str(factor_row['Name'])
                if source in IGNORE_LIST:
                    continue
                try:
                    summary, source_flare_rows = analyze_source(
                        cadence_df=cadence_df,
                        factor_row=factor_row,
                        class_sets=class_sets,
                        percent=float(percent),
                        cosi_bkg_rate=cosi_bkg_rate,
                    )
                    quiescent_rows.append(summary)
                    flare_rows.extend(source_flare_rows)
                except Exception as exc:
                    quiescent_rows.append(
                        {
                            'Name': source,
                            'quiescent_background': np.nan,
                            'quiescent_background_error': np.nan,
                            'average_flux': np.nan,
                            'average_flux_cosi_band': np.nan,
                            'thresholdflux': np.nan,
                            'points_used': 0,
                            'time_start_mjd': np.nan,
                            'time_end_mjd': np.nan,
                            'error': str(exc),
                        }
                    )

            flare_df = pd.DataFrame(flare_rows)
            if not flare_df.empty:
                flare_df = flare_df.loc[flare_df['Source_Counts(ph)_(0.2-5_MeV)'] != 0].reset_index(drop=True)
                flare_df = flare_df.loc[flare_df['Duration_(s)'] <= 2e8].reset_index(drop=True)

            quiescent_df = pd.DataFrame(quiescent_rows)
            flare_path = output_dir / f'November2025_COSI_Eta{float(percent):.1f}_bkg{float(bkg):.2f}.csv'
            quiescent_path = output_dir / f'November2025_Quiescent_Backgrounds_Eta{float(percent):.1f}_bkg{float(bkg):.2f}.csv'
            flare_df.to_csv(flare_path, index=False)
            quiescent_df.to_csv(quiescent_path, index=False)
            outputs.append(
                {
                    'percent': float(percent),
                    'background_rate': float(bkg),
                    'flare_output': str(flare_path.relative_to(ROOT)),
                    'quiescent_output': str(quiescent_path.relative_to(ROOT)),
                    'flare_rows': int(len(flare_df)),
                    'sources_processed': int(len(quiescent_df)),
                }
            )
    return outputs