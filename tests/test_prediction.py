import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import load_model, predict


VALID_SAMPLE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def test_model_can_be_loaded():
    model = load_model()

    assert model is not None
    assert "model" in model
    assert "features" in model


def test_valid_prediction():
    result = predict(VALID_SAMPLE)

    assert isinstance(result, int)
    assert result == 999


def test_missing_feature_is_rejected():
    incomplete_sample = VALID_SAMPLE.copy()
    del incomplete_sample["alcohol"]

    with pytest.raises(ValueError, match="Missing required feature"):
        predict(incomplete_sample)