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
                }
            ]
        )
    }

    text_body, html_body, inline_images, total_detections = build_multi_threshold_email_content(
        detections_by_multiplier,
        '2026-W30',
        include_potential_plots=True,
        include_confirmed_plots=False,
    )

    assert total_detections == 1
    assert 'Test Source' in text_body
    assert '12.34' in html_body
    assert '45.67' in html_body
    assert inline_images == []
