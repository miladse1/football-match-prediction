from pathlib import Path

from football_pipeline.predict import format_prediction


def test_format_prediction_percentages():
    text = format_prediction("Arsenal", "Manchester United", 0.29, 0.27, 0.44)
    assert text == (
        "Arsenal vs Manchester United\n"
        "  Home Win: 44%\n"
        "  Draw: 27%\n"
        "  Away Win: 29%"
    )


def test_predict_upsert_does_not_overwrite_played_matches():
    source = Path("src/football_pipeline/predict.py").read_text(encoding="utf-8")
    assert "played.is_played" in source
    assert "WHERE NOT EXISTS" in source
