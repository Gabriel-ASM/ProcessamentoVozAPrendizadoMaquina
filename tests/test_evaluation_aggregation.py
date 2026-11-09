from voice_localization.evaluation import _summarize
import pytest


def test_summary_uses_all_source_errors_instead_of_averaging_scene_percentiles():
    rows = [
        {"mean_error_m": 1.0, "source_errors_m": [1.0], "n_sources": 1},
        {"mean_error_m": 0.0, "source_errors_m": [0.0, 0.0, 0.0], "n_sources": 3},
    ]
    result = _summarize(rows)
    assert result["source_count"] == 4
    assert result["mean_error_m"] == 0.25
    assert result["median_error_m"] == 0.0
    assert result["p90_error_m"] == pytest.approx(0.7)
    assert result["within_tolerance"] == 0.75
    assert result["by_n_sources"]["1"]["within_tolerance"] == 0.0
    assert result["by_n_sources"]["3"]["within_tolerance"] == 1.0
