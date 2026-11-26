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

## Important limitations

- The Pima dataset is small and not representative of all populations. Its results
  are for learning and must not be used to diagnose or guide treatment.
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
- The current step does not yet include accounts, a database, an API, or Docker.
  Those should be added after the model and its evaluation are understood.
