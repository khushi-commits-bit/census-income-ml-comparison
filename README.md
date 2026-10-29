# Census Income Classification: 10-Model Comparative Study

Predict whether annual income exceeds $50,000 using UCI Adult data. This is an
educational historical-data benchmark, not an income assessment or hiring tool.
No benchmark results are claimed before the code is run on the real dataset.

## Windows + VS Code setup

Use Python 3.11 or 3.12. Extract this project, open its folder in VS Code,
and open Terminal > New Terminal. Run one command at a time:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py --quick --folds 3
```

Using the virtual environment's executable directly avoids PowerShell activation
policy errors. The first run needs internet to download the dataset. The quick
run uses 3,000 training and 1,500 test rows and is only a demonstration.

For the full study (may take a substantial amount of time on a laptop):

```powershell
.\.venv\Scripts\python.exe src/train.py
```

Optional: tune the two highest-ranked non-baseline models with three parameter
settings each, selected and evaluated exclusively within training CV:

```powershell
.\.venv\Scripts\python.exe src/train.py --tune
```

Do not use quick results as full-dataset findings. Re-running overwrites files
in results/quick or results/full. Use the same seed for a reproducible comparison.
Default --jobs 1 limits memory pressure. On macOS/Linux use .venv/bin/python.

## Models

Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees,
Gaussian Naive Bayes, RBF SVM, AdaBoost, Gradient Boosting, and Histogram
Gradient Boosting, plus a majority-class baseline.

## Methodology

- Preserve UCI's official training/test partition. Remove exact full-record
  duplicates within partitions and training-overlapping full records from test.
  This cannot guarantee that different records represent different individuals.
- Exclude fnlwgt (survey weight) from predictors; metrics are unweighted.
  Exclude education text because education_num describes the same concept.
- Fit median/mode imputers, scaling, and dense one-hot encoding inside every
  cross-validation fold. Unseen categorical values are ignored by the encoder.
- Reuse identical shuffled stratified folds for all models. Rank by mean F1
  for the >50K class. Save fold SD; it is not a confidence interval or a test
  of statistical significance. Defaults are practical configurations, not
  claims of optimal settings. A common encoding favors some models over others.
- Optional top-two tuning uses a small equal candidate count, not equal compute.
  Its best CV scores have selection optimism; the independent test is the final
  evaluation. This is not nested cross-validation.
- Freeze the selected model before test evaluation. Report test performance of
  all models descriptively. Do not choose a replacement winner from test results.
- Use probabilities where available and SVM decision scores otherwise for AUC
  and precision-recall curves. Default classification thresholds are unchanged.
  No misleading shared training-loss comparison is produced.

## Outputs

Under results/quick or results/full:

| File | Meaning |
| --- | --- |
| cv_results.csv, cv_folds.csv | Mean/SD metrics and individual fold results |
| test_results.csv | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision and runtime |
| cv_comparison.png | CV F1 with fold SD |
| roc_curves.png, precision_recall_curves.png | Held-out discrimination comparisons |
| confusion_matrix.png | Selected model errors |
| income_distribution.png, hours_by_income.png | Training-only exploratory charts |
| feature_importance.csv/.png | Original-column permutation importance for selected model |
| subgroup_metrics.csv | Selected model sex/race subgroup diagnostics |
| misclassifications.csv | Records the selected model classified incorrectly |
| run_metadata.json | Seed, versions, parameters, row counts, raw-file checksums |
| best_model.joblib | Complete selected preprocessing/model pipeline |

Never load a joblib model from an untrusted source. The saved pipeline expects
the same raw feature names and types used by this script.

## Write your findings after running

1. Which model had the highest training CV F1? How variable were its folds?
2. How far ahead was it of the majority baseline on the held-out test?
3. Did extra training time produce a useful improvement?
4. Which errors and feature relationships stood out?
5. How did recall and false-positive rates vary across subgroups?

Subgroup metrics are descriptive, not proof of fairness. Small groups can give
unstable estimates; inspect sample and positive counts. Demographic features
are retained for study, and removing them alone would not remove bias.
Permutation importance is model-specific, not causal, and correlated columns
can share importance. Historical US data and this fixed income threshold limit
present-day and international relevance. Do not report causal conclusions.

## Upload to GitHub

Create an empty repository named census-income-ml-comparison on GitHub.
Do not initialize it with a README. Open the terminal in this project folder:

```powershell
git init
git add .
git commit -m "Add reproducible census income model comparison"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/census-income-ml-comparison.git
git push -u origin main
```

Replace YOUR_USERNAME. After a successful full run, add your interpretation and
link selected charts in this README, then commit and push the results.
For an exact dependency snapshot after a successful run:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

## References

- UCI Adult: https://archive.ics.uci.edu/dataset/2/adult
- Scikit-learn pipelines: https://scikit-learn.org/stable/modules/compose.html
- Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
