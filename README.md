# Diabetes Risk Prediction

Comparing a Multi-Layer Perceptron and a Support Vector Machine for diabetes screening on imbalanced public health data, with a focus on how class balancing changes recall rather than ranking.

## Overview

This project predicts diabetes risk from the BRFSS 2015 Diabetes Health Indicators dataset using two neural computing methods, a Multi-Layer Perceptron (MLP) built in PyTorch and a Support Vector Machine (SVM) from scikit-learn. The central question is practical rather than purely predictive. In a screening setting, missing a positive case is more costly than a false alarm. The project therefore focuses on how class-balancing choices affect a model's ability to catch positive cases at the default decision threshold.

## Data

* **Source:** BRFSS 2015 Diabetes Health Indicators (Kaggle)
* **Sample:** a stratified sample of 100,000 observations with 21 health indicators
* **Target:** the original three-class label was converted into a binary target, healthy (0) versus pre-diabetes or diabetes (1), for screening-oriented classification
* **Class balance:** approximately 15.7% positive, an imbalance handled explicitly
* **Split:** 80% training and 20% test, with standardisation fitted on the training data only to avoid leakage

The raw dataset is not included in this repository. To run the notebook, download the data from the Kaggle page and provide it at the path the notebook expects.

## Approach

Three models are trained and compared:

1. **MLP without SMOTE.** A two-hidden-layer MLP (ReLU activations) trained on the imbalanced data as a baseline.
2. **MLP with SMOTE.** The same architecture, with SMOTE oversampling applied only to the training folds to address imbalance without leaking into validation or test data.
3. **Cost-sensitive SVM.** Linear and RBF kernels, with class weighting to reflect the cost of missing positive cases.

Key methodology:

* Hyperparameter selection for the MLP used 5-fold cross-validation with early stopping. The SVM was tuned on a single stratified hold-out split because full cross-validation was not computationally feasible on CPU, and this asymmetry is documented as a limitation.
* SMOTE was applied inside training folds only, never on validation or test data.
* The test set was held out and untouched during all model selection.
* ROC-AUC was the primary selection metric because accuracy is misleading under imbalance. Recall, precision, F1, and confusion matrices were also reported to capture threshold-dependent behaviour.

## Results

| Model | ROC-AUC (test) | Recall (positive class) |
|---|---|---|
| MLP without SMOTE | 0.818 | 0.079 |
| MLP with SMOTE | 0.815 | 0.765 |
| Cost-sensitive SVM | 0.813 | 0.781 |

The three models reached almost identical ROC-AUC, yet their behaviour at the default threshold was completely different. The baseline MLP, despite the highest AUC, detected almost no positive cases (recall 0.079). Adding SMOTE left AUC essentially unchanged but raised recall to 0.765, and the cost-sensitive SVM behaved similarly (recall 0.781). This higher recall came at the cost of lower precision, which is an acceptable trade-off in a screening context where false negatives carry greater cost than false positives.

The main lesson is that similar ranking performance (AUC) can hide very different real-world behaviour at a fixed threshold, which is why metric choice must match the objective.

## Tech stack

Python, PyTorch, scikit-learn, imbalanced-learn (SMOTE), NumPy, pandas, matplotlib, joblib.

## Files

* `diabetes-prediction.ipynb` is the full analysis notebook, including code, figures, and results.

## Notes

This project was developed during the Neural Computing module of the MSc Data Science programme at City St George's, University of London. It applies neural networks, support vector machines, class-imbalance handling, and threshold-aware evaluation, themes that transfer directly to risk classification problems where catching positive cases matters more than overall accuracy.
