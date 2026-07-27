import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PythonFiles'))

from PythonFiles import notebook_pipeline

import AutomatedScript
from AutomatedScript import build_incremental_summary_for_source, build_multi_threshold_email_content, plot_light_curve


def test_build_multi_threshold_email_content_uses_section_specific_mdp_values() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Test Source',
                    'potential_points': 1,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': 99.99,
                    'latest_potential_mdp99_percent': 90.0,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': 'missing_plot.png',
                    'confirmed_flare_plot': 'missing_plot.png',
                    'latest_new_point_flux_cosi': 1.2e-7,
                    'latest_flare_flux_threshold': 5e-8,
                },
                {
                    'Name': 'Second Source',
                    'potential_points': 2,
                    'active_flare_weeks': 1,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': True,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': True,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': 78.91,
                    'latest_potential_mdp99_percent': 70.0,
                    'latest_active_mdp99_percent': 60.0,
                    'peak_potential_flux': 2.34e-7,
                    'mean_potential_flux': 2.20e-7,
                    'latest_threshold_cosi_flux': 9.0e-8,
                    'latest_confirmed_sigma_delta': 2.2,
                    'potential_flare_plot': 'missing_plot.png',
                    'confirmed_flare_plot': 'missing_plot.png',
                    'latest_new_point_flux_cosi': 2.3e-7,
                    'latest_flare_flux_threshold': 6e-8,
                },
            ]
        )
    }

    text_body, html_body, inline_images, total_detections = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=True,
        include_confirmed_plots=False,
    )

    assert total_detections == 2
    assert 'Test Source' in text_body
    assert 'Second Source' in text_body
    assert '90.00' in html_body
    assert '60.00' in html_body
    assert html_body.count('<tr') >= 4
    assert inline_images == []


def test_build_multi_threshold_email_content_falls_back_to_counts_and_latest_mdp() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Fallback Source',
                    'potential_points': 2,
                    'active_flare_weeks': 1,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': False,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': False,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': 88.88,
                    'latest_potential_mdp99_percent': np.nan,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': np.nan,
                    'mean_potential_flux': np.nan,
                    'latest_threshold_cosi_flux': np.nan,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': np.nan,
                    'latest_flare_flux_threshold': np.nan,
                }
            ]
        )
    }

    text_body, html_body, inline_images, total_detections = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=False,
        include_confirmed_plots=False,
    )

    assert total_detections == 1
    assert 'Fallback Source' in text_body
    assert 'Yes' in html_body
    assert '88.88' in html_body
    assert inline_images == []


def test_plot_light_curve_supports_incremental_scan_columns(tmp_path: Path) -> None:
    output_path = tmp_path / 'incremental_plot.png'
    dataframe = pd.DataFrame(
        {
            'new_point_mjd': [60000.0, 60007.0],
            'new_point_flux': [1.0e-7, 1.5e-7],
            'new_point_flux_error': [1.0e-8, 1.2e-8],
        }
    )

    plot_light_curve(dataframe, 'Test Source', output_path, title_suffix='incremental test')

    assert output_path.exists()


def test_incremental_flare_scan_marks_threshold_crossings_as_potential_flare_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(notebook_pipeline, 'load_weekly_database', lambda *args, **kwargs: pd.DataFrame({'source_name': ['Test Source'], 'photon_flux2': [1.0], 'photon_flux_error2': [0.1], 'tmin': [0], 'cadence': ['weekly']}))
    monkeypatch.setattr(notebook_pipeline, 'build_source_arrays', lambda *args, **kwargs: (None, np.array([60000.0, 60007.0, 60014.0]), np.array([1.0e-7, 3.0e-7, 1.0e-7]), np.array([1.0e-8, 1.0e-8, 1.0e-8]), 1.0e-7))
    monkeypatch.setattr(notebook_pipeline, 'resolve_quiescent_background', lambda **kwargs: {'quiescent_background': 1.0e-7, 'quiescent_background_error': 1.0e-8, 'detection_threshold': 1.0e-7, 'origin': 'cache', 'cache_path': str(tmp_path / 'qb.csv')})

    result = notebook_pipeline.incremental_flare_scan(
        source_name='Test Source',
        database_path='db.csv',
        percent=0.3,
        cadence='weekly',
        lookback_weeks=4.0,
        detection_method='sigma',
        flare_threshold_multiplier=2.0,
        confirmed_sigma_threshold=2.0,
        consecutive_points=1,
        cache_path=str(tmp_path / 'qb_cache.csv'),
        source_json_path=None,
    )

    assert result['potential_flare_point'].tolist() == [True, False]
    assert result['flare_active'].tolist() == [False, False]


def test_build_incremental_summary_for_source_builds_summary_before_writing_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = pd.DataFrame(
        [
            {
                'flare_active': False,
                'potential_flare_point': False,
                'confirmed_flare_active': False,
                'quiescent_background_origin': 'cache',
                'quiescent_background_cache_path': str(tmp_path / 'qb.csv'),
                'new_point_mjd': 60000.0,
                'new_point_flux': 1.0e-7,
                'flare_flux_threshold': 1.0e-7,
                'average_flux_full_series': 8e-8,
                'mdp99_percent': 12.34,
                'mdp99_available': True,
                'confirmed_sigma_delta': 1.5,
                'flare_start_mjd': 60000.0,
                'confirmed_flare_start_mjd': 60000.0,
            }
        ]
    )

    monkeypatch.setattr(AutomatedScript.notebook_pipeline, 'incremental_flare_scan', lambda **kwargs: result.copy())
    monkeypatch.setattr(AutomatedScript, 'add_incremental_mdp99_columns', lambda frame, factor_row, **kwargs: frame)
    monkeypatch.setattr(AutomatedScript, 'plot_light_curve', lambda *args, **kwargs: None)
    monkeypatch.setattr(AutomatedScript, 'write_source_json_output', lambda **kwargs: None)

    args = Namespace(
        flare_threshold_multiplier=1.0,
        db_path='db.csv',
        incremental_percent=10.0,
        lightcurve_cadence='weekly',
        lookback_weeks=4.0,
        detection_method='sigma',
        confirmed_sigma_threshold=3.0,
        consecutive_points=1,
        qb_cache_path=str(tmp_path / 'qb_cache.csv'),
        cosi_background_rate=0.0,
        arm_reduction=1.0,
        mdp99_average_mu=1.0,
    )

    summary_row, returned_result = build_incremental_summary_for_source(
        args,
        'Test Source',
        pd.DataFrame([{'Name': 'Test Source', 'Int_flux_ratio': 1.0}]),
    )

    assert summary_row['Name'] == 'Test Source'
    assert returned_result.equals(result)
    assert summary_row['latest_new_point_flux'] == pytest.approx(1.0e-7)


def test_build_multi_threshold_email_content_splits_potential_and_active_rows_by_state() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Potential Only',
                    'potential_points': 1,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': np.nan,
                    'latest_potential_mdp99_percent': 90.0,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': 1.2e-7,
                    'latest_flare_flux_threshold': 5e-8,
                },
                {
                    'Name': 'Active Source',
                    'potential_points': 2,
                    'active_flare_weeks': 1,
                    'confirmed_flare_weeks': 1,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': True,
                    'had_confirmed_flare_weeks': True,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': True,
                    'latest_confirmed_flare_active': True,
                    'latest_mdp99_percent': np.nan,
                    'latest_potential_mdp99_percent': np.nan,
                    'latest_active_mdp99_percent': 80.0,
                    'peak_potential_flux': 2.34e-7,
                    'mean_potential_flux': 2.20e-7,
                    'latest_threshold_cosi_flux': 9.0e-8,
                    'latest_confirmed_sigma_delta': 2.2,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': 2.3e-7,
                    'latest_flare_flux_threshold': 6e-8,
                },
            ]
        )
    }

    _, html_body, _, _ = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=False,
        include_confirmed_plots=False,
    )

    assert html_body.count('Potential Only') == 1
    assert html_body.count('Active Source') >= 1
    assert '90.00' in html_body
    assert '80.00' in html_body


def test_build_multi_threshold_email_content_excludes_confirmed_rows_from_potential_section() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Confirmed Not Currently Active',
                    'potential_points': 2,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 1,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': True,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': True,
                    'latest_mdp99_percent': np.nan,
                    'latest_potential_mdp99_percent': 90.0,
                    'latest_active_mdp99_percent': 80.0,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': 2.5,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': 1.2e-7,
                    'latest_flare_flux_threshold': 5e-8,
                },
            ]
        )
    }

    _, html_body, _, _ = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=False,
        include_confirmed_plots=False,
    )

    assert html_body.count('Confirmed Not Currently Active') == 1
    potential_table, confirmed_table = html_body.split('Confirmed flare detections', 1)
    assert 'Confirmed Not Currently Active' not in potential_table
    assert 'Confirmed Not Currently Active' in confirmed_table


def test_build_multi_threshold_email_content_filters_sources_with_mdp_above_100_percent() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Filtered Source',
                    'potential_points': 1,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': 120.0,
                    'latest_potential_mdp99_percent': 120.0,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': 1.2e-7,
                    'latest_flare_flux_threshold': 5e-8,
                },
                {
                    'Name': 'Included Source',
                    'potential_points': 1,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': 80.0,
                    'latest_potential_mdp99_percent': 80.0,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': 1.2e-7,
                    'latest_flare_flux_threshold': 5e-8,
                },
            ]
        )
    }

    text_body, html_body, _, _ = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=False,
        include_confirmed_plots=False,
    )

    assert 'Filtered Source' not in html_body
    assert 'Included Source' in html_body
    assert 'Filtered Source' not in text_body
    assert 'Included Source' in text_body


def test_build_multi_threshold_email_content_uses_saved_source_json_summary(tmp_path: Path) -> None:
    saved_json_path = tmp_path / 'saved_summary.json'
    saved_json_path.write_text(
        json.dumps(
            {
                '_scanState': {
                    'emailSummary': {
                        'potentialPoints': 3,
                        'activeFlareWeeks': 2,
                        'confirmedFlareWeeks': 1,
                        'latestPotentialFlarePoint': True,
                        'latestFlareActive': True,
                        'latestConfirmedFlareActive': True,
                        'latestMdp99Percent': 77.7,
                        'latestPotentialMdp99Percent': 90.0,
                        'latestActiveMdp99Percent': 80.0,
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Saved JSON Source',
                    'potential_points': 0,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 0,
                    'had_potential_flare_points': False,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': False,
                    'latest_potential_flare_point': False,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': False,
                    'latest_mdp99_percent': np.nan,
                    'latest_potential_mdp99_percent': np.nan,
                    'latest_active_mdp99_percent': np.nan,
                    'peak_potential_flux': np.nan,
                    'mean_potential_flux': np.nan,
                    'latest_threshold_cosi_flux': np.nan,
                    'latest_confirmed_sigma_delta': np.nan,
                    'potential_flare_plot': '',
                    'confirmed_flare_plot': '',
                    'latest_new_point_flux_cosi': np.nan,
                    'latest_flare_flux_threshold': np.nan,
                    'source_json': str(saved_json_path),
                }
            ]
        )
    }

    text_body, html_body, _, _ = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=False,
        include_confirmed_plots=False,
    )

    assert 'Saved JSON Source' in html_body
    assert 'Yes' in html_body
    assert '77.7' in text_body
    assert '110.00' not in html_body
    assert '120.00' not in html_body
