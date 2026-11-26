# diabetese_pred

The default training workflow uses a small tabular logistic-regression model.
The earlier neural-network experiment remains in `model_pipeline.py` for
comparison, but it is not the recommended model for this dataset.

## Train and compare models

Use Python 3.11 or newer, create a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
```

The default script reads `pima-indians-diabetes-labeled.csv`, a named-column
copy of the original headerless `pima-indians-diabetes.csv`. It replaces invalid
zeros with missing values, then fits median imputation and standard scaling
inside each training fold. It uses stratified 5-fold cross-validation and
reports accuracy, precision, recall, and ROC AUC. The final model and metrics
are saved under `artifacts/simple_model/`.

The prior neural-network experiment compares imputation strategies and batch
sizes. It is more complex and did not outperform the simpler unweighted logistic
regression on cross-validation precision and accuracy.

## Run the local demo app

The Streamlit app loads the saved logistic-regression pipeline, lets a user create
a local account, makes a prediction from entered measurements, and saves that
user's prediction history in a local SQLite database. Passwords are stored as
salted hashes. The database is excluded from Git.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. For measurements that are not available, enter `0`
for glucose, blood pressure, skin thickness, insulin, or BMI; the model treats
those values as missing and imputes them. Do not enter names or other identifying
information.

## Run with Docker

Install Docker Desktop, then start the app from this directory:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. Compose binds the app to localhost and persists the
SQLite database in a Docker-managed volume. Stop it with `Ctrl+C`, then run
`docker compose down` when finished.

To run the older neural-network experiment, install its optional dependency set
with `python -m pip install -r requirements-neural.txt`.

## Important limitations

- The Pima dataset is small and not representative of all populations. Its results
  are for learning and must not be used to diagnose or guide treatment.
- The app is a local educational demo, not a medical product. Do not use it for
  real clinical decisions or enter identifiable or real patient information.
  Its database is not encrypted at rest; account passwords do not make health
  history storage production-secure.
- Accuracy and precision depend on the decision threshold and class balance.
  Evaluate recall, precision, calibration, and the costs of false positives and
  false negatives before choosing a threshold for any application.
- The default model does not use class weights: this improves precision in the
  current cross-validation comparison, but reduces recall, so it will miss more
  positive cases than a recall-oriented model.
- These rows are independent tabular measurements, not sequences. Dropout and
  early stopping can help control overfitting in a neural network, but do not
  guarantee better accuracy or precision. An LSTM requires ordered measurements
  over time; reinforcement learning does not fit this supervised prediction task.
- Class-wise imputation is intentionally not used: a prediction-time system
  cannot know the true outcome label needed to select the class median. Glucose
  bins are exploratory, and standard clinical cutoffs assume a measurement
  context (such as fasting status) that this dataset does not establish.
