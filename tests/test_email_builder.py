import json
from pathlib import Path

import numpy as np
import pandas as pd

from AutomatedScript import build_multi_threshold_email_content


def test_build_multi_threshold_email_content_uses_section_specific_mdp_values() -> None:
    detections_by_multiplier = {
        3.0: pd.DataFrame(
            [
                {
                    'Name': 'Test Source',
                    'potential_points': 1,
                    'active_flare_weeks': 0,
                    'confirmed_flare_weeks': 1,
                    'had_potential_flare_points': True,
                    'had_active_flare_weeks': False,
                    'had_confirmed_flare_weeks': True,
                    'latest_potential_flare_point': True,
                    'latest_flare_active': False,
                    'latest_confirmed_flare_active': True,
                    'latest_mdp99_percent': 99.99,
                    'latest_potential_mdp99_percent': 12.34,
                    'latest_active_mdp99_percent': 45.67,
                    'peak_potential_flux': 1.23e-7,
                    'mean_potential_flux': 1.10e-7,
                    'latest_threshold_cosi_flux': 8.0e-8,
                    'latest_confirmed_sigma_delta': 4.1,
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
                    'latest_potential_mdp99_percent': 23.45,
                    'latest_active_mdp99_percent': 56.78,
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
    assert '12.34' in html_body
    assert '45.67' in html_body
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
                        'latestPotentialMdp99Percent': 11.11,
                        'latestActiveMdp99Percent': 22.22,
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
    assert '11.11' in html_body
    assert '22.22' in html_body
