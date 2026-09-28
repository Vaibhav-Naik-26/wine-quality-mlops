# Automating an ML Pipeline with GitHub Actions

## 1. Project Overview

This project implements an automated machine learning pipeline using GitHub Actions.

The pipeline automatically:

1. Downloads and validates the dataset.
2. Splits the data into training and validation sets.
3. Trains a baseline model.
4. Trains a candidate machine learning model.
5. Evaluates the candidate against the baseline.
6. Applies a fixed quality gate.
7. Runs application tests.
8. Creates and publishes a model package only when all checks pass.

The pipeline runs automatically whenever changes are pushed to the `main` branch.

### Setup Instructions

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/Vaibhav-Naik-26/wine-quality-mlops.git
cd wine-quality-mlops
pip install -r requirements.txt
```

To train and evaluate the model locally:

```bash
python src/train.py
```

If the quality gate passes, the trained model and metrics are created in the `artifacts/` directory.

To run the application tests:

```bash
python -m pytest -q
```

The GitHub Actions workflow performs these steps automatically in a clean Ubuntu environment, so the pipeline does not depend on the local development environment.

---

## 2. Dataset

### Dataset

Wine Quality - Red Wine dataset.

### Dataset source

The dataset is obtained from the UCI Machine Learning Repository.

Source:

https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

The pipeline downloads the dataset directly from the public URL, so the dataset does not need to be stored in the repository.

### Prediction task

This project performs binary classification.

The original `quality` value is converted into a binary target:

- Quality >= 6 → 1 (good quality)
- Quality < 6 → 0 (not good quality)

### Target variable

`target`

### Input features

The model uses the following 11 input features:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

---

## 3. Evaluation Metric

The evaluation metric used is **F1-score**.

F1-score is suitable for this classification task because it combines precision and recall into one metric.

The dataset contains two classes, and F1-score provides a better view of classification performance than accuracy alone when the classes are not perfectly balanced.

The same metric is used for both the baseline and candidate model.

---

## 4. Training and Validation

The dataset is divided into:

- 80% training data
- 20% validation data

The split uses:

`random_state = 42`

This makes the experiment reproducible.

The validation data is kept separate from model training.

Preprocessing is not required for this dataset because the selected Random Forest model can work directly with the numeric input features.

---

## 5. Baseline Model

The baseline model is:

`DummyClassifier(strategy="most_frequent")`

The baseline F1-score was:

**0.6965**

The baseline provides a simple reference point for evaluating whether the candidate model provides a meaningful improvement.

---

## 6. Candidate Model

The candidate model is:

`RandomForestClassifier`

Configuration:

- `n_estimators = 200`
- `random_state = 42`
- `n_jobs = -1`

The candidate model achieved an F1-score of:

**0.8249**

---

## 7. Quality Gate

The improvement margin is:

**0.05 F1-score**

The quality gate is:

`candidate F1 >= baseline F1 + 0.05`

The baseline score was:

`0.6965`

Therefore the minimum required candidate score was:

`0.7465`

The actual candidate score was:

`0.8249`

Therefore the quality gate passed.

### Why was a 0.05 margin chosen?

A margin of 0.05 requires a meaningful improvement instead of allowing the candidate model to pass because of a very small difference from the baseline.

If the margin were too low, a model with only a tiny improvement could pass the quality gate.

If the margin were too high, a useful model could be rejected even when it provides a meaningful improvement.

The margin is kept unchanged during the failure and recovery demonstrations.

---

## 8. Validation Checks

The training script validates that:

- The required `quality` column exists in the downloaded dataset.
- Required feature columns exist.
- The generated target column exists.
- Required values do not contain missing data.
- The dataset contains enough rows.

If validation fails, the script raises an exception and the workflow receives a non-zero exit code.

The quality gate also causes the training step to return a non-zero exit code when the candidate model does not meet the required improvement.

---

## 9. Application Tests

The project contains automated tests in:

`tests/test_prediction.py`

The tests verify:

### Test 1 - Model loading

The saved model can be loaded successfully.

### Test 2 - Valid prediction

A valid input sample produces an integer prediction of either 0 or 1.

### Test 3 - Missing feature

An input missing a required feature is rejected with a clear error.

All three tests pass in the final pipeline.

---

## 10. GitHub Actions Workflow

The workflow is located at:

`.github/workflows/ml-pipeline.yml`

The workflow runs on every push to the `main` branch.

The pipeline performs:

```text
Checkout repository
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Train and evaluate model
        ↓
Run application tests
        ↓
Create model package
        ↓
Upload artifact
```

Training, evaluation, application testing, and artifact creation are performed in the same GitHub Actions job.

The artifact upload step is reached only when the previous steps have completed successfully.

The workflow uses Python scripts rather than notebooks and does not depend on Colab, Google Drive, or the local development environment.

---

## 11. Workflow Evidence

The following GitHub Actions runs demonstrate the pipeline behavior during development and recovery.

### Workflow Links

- [Run #1 - Initial successful pipeline](https://github.com/Vaibhav-Naik-26/wine-quality-mlops/actions/runs/35987903277)
- [Run #2 - Failure A - Model quality gate](https://github.com/Vaibhav-Naik-26/wine-quality-mlops/actions/runs/35990554662)
- [Run #3 - Recovery from Failure A](https://github.com/Vaibhav-Naik-26/wine-quality-mlops/actions/runs/35993021085)
- [Run #4 - Failure B - Application test](https://github.com/Vaibhav-Naik-26/wine-quality-mlops/actions/runs/35993416094)
- [Run #5 - Final successful pipeline](https://github.com/Vaibhav-Naik-26/wine-quality-mlops/actions/runs/35993646526)

### Failure A - Model Quality Gate

The first deliberate failure was caused by changing the candidate Random Forest configuration to a weaker model.

The candidate model was temporarily changed to:

`RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)`

This resulted in a candidate F1-score below the required quality gate.

The baseline F1-score was:

`0.6965`

The candidate F1-score was:

`0.6827`

The minimum required F1-score was:

`0.7465`

Therefore, the quality gate failed.

The workflow stopped during the training and evaluation step and no model artifact was published.

### Recovery from Failure A

The candidate Random Forest configuration was restored to the working configuration:

- `n_estimators = 200`
- `random_state = 42`
- `n_jobs = -1`

The pipeline was then run again.

The candidate achieved an F1-score of:

`0.8249`

The quality gate passed and the workflow completed successfully.

A model artifact was published in the successful recovery run.

### Failure B - Application Test

The second deliberate failure was created by changing an application test so that it expected an incorrect prediction value.

The model quality gate still passed, but the application test failed.

Because the test step returned a non-zero exit code, the workflow stopped before the artifact creation and upload step.

Therefore, no model artifact was published from the failed run.

### Recovery from Failure B

The incorrect test assertion was restored to the correct application test.

The pipeline was executed again.

The model quality gate passed.

All three application tests passed.

The model package was then created and uploaded successfully.

---

## 12. Model Package and Artifact

The model package is created only after training, quality evaluation, and application tests have passed.

The package contains:

```text
model.joblib
metrics.json
predict.py
requirements.txt
```

### `model.joblib`

Contains the trained Random Forest model and the list of expected input features.

### `metrics.json`

Contains the baseline score, candidate score, required score, margin, and gate result.

### `predict.py`

Contains the application prediction logic used to load the model and make predictions.

### `requirements.txt`

Contains the Python dependencies required to use the model package.

The final successful pipeline produced:

`model-package-5`

The artifact was downloaded and verified after the successful run.

The artifact is identified using the GitHub Actions workflow run number.

---

## 13. CI and Artifact Delivery

The pipeline demonstrates continuous integration because changes pushed to the `main` branch automatically trigger the GitHub Actions workflow.

The workflow checks out the repository, creates a clean Python environment, installs dependencies, trains and evaluates the model, and runs the application tests automatically.

Artifact delivery is demonstrated by creating a model package and uploading it with `actions/upload-artifact@v4`.

The artifact upload step does not run when an earlier training or testing step fails.

This prevents an invalid or untested model from being published.

The final artifact therefore represents a model that has passed both the model quality gate and the application tests.

---

## 14. Failure and Recovery Summary

Two deliberate failures were introduced to demonstrate that the pipeline can detect different types of problems.

| Run | Situation | Result | Artifact |
|---|---|---|---|
| Run #1 | Initial pipeline | Passed | Published |
| Run #2 | Candidate model failed quality gate | Failed | Not published |
| Run #3 | Quality gate issue corrected | Passed | Published |
| Run #4 | Application test deliberately failed | Failed | Not published |
| Run #5 | Application test corrected | Passed | Published |

This demonstrates that the pipeline does not publish an artifact when either the model quality check or application testing fails.

---

## 15. MLOps Maturity

This project represents an early-stage automated MLOps workflow, corresponding to a basic Level 1-style pipeline automation stage.

The implementation includes:

- Automated data validation
- Reproducible training
- Baseline comparison
- Candidate model evaluation
- Automated quality gating
- Automated application testing
- Continuous integration using GitHub Actions
- Conditional model packaging
- Artifact delivery

The workflow automates the training and validation process, but it does not include production deployment, continuous production monitoring, automated model retraining, or data/model drift monitoring.

Possible improvements for the next maturity stage include:

- Experiment tracking
- Model versioning
- Data versioning
- Automated deployment
- Scheduled retraining
- Production performance monitoring
- Automated data drift detection
- Model performance monitoring after deployment

---

## 16. Repository Structure

The main project files are organized as follows:

```text
wine-quality-mlops/
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `artifacts/` directory is generated during training and is excluded from Git using `.gitignore`.

---

## 17. Brief Answers to Required Questions

### 1. Why does your evaluation metric suit your task?

F1-score is suitable because this is a binary classification problem and F1 combines precision and recall into a single metric. It provides a useful measure of classification performance when the class distribution is not perfectly balanced.

### 2. Why did you choose this improvement margin? What would happen if it were too low or too high?

A margin of `0.05 F1-score` was chosen to require a meaningful improvement over the baseline. If the margin were too low, a very small improvement could allow a candidate model to pass. If it were too high, a useful model could be rejected even when it provides a meaningful improvement.

### 3. What caused each failed run? Which check prevented publication?

**Failure A:** The Random Forest configuration was deliberately weakened, producing an F1-score of `0.6827`, below the required `0.7465`. The model quality gate prevented publication.

**Failure B:** An application test was deliberately changed to expect an incorrect prediction. The model quality gate passed, but the application test failed. The failed test prevented publication.

The model configuration and application test were then restored, resulting in a successful final run.

### 4. Which parts of your workflow demonstrate continuous integration and artifact delivery?

Continuous integration is demonstrated by the GitHub Actions workflow automatically running whenever changes are pushed to `main`. It creates a clean Python environment, installs dependencies, trains and evaluates the model, and runs automated tests.

Artifact delivery is demonstrated by creating the model package and uploading it using `actions/upload-artifact@v4`. The package is uploaded only after all preceding checks pass.

### 5. Which MLOps maturity level best describes your implementation? Justify your answer using its actual capabilities, and identify what remains to reach the next level.

This project represents an early-stage automated MLOps workflow, corresponding to a basic Level 1-style pipeline automation stage. It has automated training, evaluation, quality gating, application testing, CI, and artifact delivery.

It does not yet include production deployment, production monitoring, automated retraining, or data/model drift monitoring. These capabilities would be needed for a more mature MLOps implementation.

---

## 18. Conclusion

This project demonstrates how a machine learning workflow can be automated using GitHub Actions.

Instead of manually training and checking the model, the pipeline automatically:

- Validates the dataset.
- Splits the data.
- Trains a baseline model.
- Trains a candidate model.
- Compares the candidate against the baseline.
- Applies a fixed quality gate.
- Runs application tests.
- Creates a model package.
- Publishes the artifact only when all required checks pass.

The deliberate failure demonstrations show that the pipeline can detect both model-quality problems and application-level problems before an artifact is published.

The final successful run confirms that the complete pipeline works from training through testing and artifact delivery.
