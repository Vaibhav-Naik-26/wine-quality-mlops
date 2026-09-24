import joblib
import pandas as pd


MODEL_PATH = "artifacts/model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict(sample):
    bundle = load_model()

    model = bundle["model"]
    features = bundle["features"]

    if not isinstance(sample, dict):
        raise TypeError("Input must be a dictionary.")

    missing = [feature for feature in features if feature not in sample]

    if missing:
        raise ValueError(
            f"Missing required feature(s): {missing}"
        )

    data = pd.DataFrame(
        [[sample[feature] for feature in features]],
        columns=features,
    )

    prediction = model.predict(data)

    return int(prediction[0])