# lone-data-
This project contains:
- EDA, data cleaning, feature engineering
- Classical ML: XGBoost
- Deep Learning: Keras ANN and PyTorch MLP
- Scripts / Jupyter Notebook cells and saved artifacts

## File list
- `Loan_Project_Consolidated.ipynb` — consolidated notebook (or paste cells into a new notebook)
- `preprocessor.joblib` — saved preprocessing pipeline (created after running notebook)
- `xgb_model.json`, `keras_ann.h5`, `pytorch_mlp.pt` — saved models (created after running)
- `requirements.txt` — Python package list
- `FINAL_REPORT.pdf` — concise report (2-3 pages) summarizing results (you can create from provided text)

- Reproducibility notes

Random seed defined as RANDOM_SEED = 42. Results may vary slightly by hardware (GPU/CPU), package versions.
Preprocessing pipeline saved to preprocessor.joblib. Use the same preprocessor at inference.
For offline-RL evaluation, the notebook uses a naive policy-value estimator (average realized reward). For robust evaluation, use IPS/DR estimators.
What to inspect

XGBoost metrics and feature importance. ANN training curves & test AUC.

Notes / Next steps

Consider temporal splits (train on older loans, test on newer loans) to avoid data leakage.
Consider using d3rlpy for a formal offline RL algorithm (CQL/AWAC) and proper off-policy evaluation tools.

Requirements.txt¶
# numpy>=1.21
# pandas>=1.3
# scikit-learn>=1.0
# matplotlib>=3.4
# seaborn>=0.11
# xgboost>=1.5
# tensorflow>=2.9
# torch>=1.12
# joblib>=1.1
# nbformat>=5.0

Loan Default Prediction — Final Report
=====================================

Author: Pranay Kumbhare
Date: 16 sept 2025

1. Project summary
------------------
This project analyzes a loan dataset to predict loan default (binary: Fully Paid = 0, Default/Charged Off = 1) and to learn an offline policy (approve/deny) using a contextual-bandit style reward model. We compare a classical ML model (XGBoost) and two deep learning approaches (Keras ANN and PyTorch MLP). 

2. Data & preprocessing
-----------------------
Dataset: (user-provided CSV)

Key preprocessing steps:
- Normalized column names (lowercase, underscores).
- Parsed common fields: `int_rate` normalized from '13.56%' to 0.1356; `term` parsed to numeric months; `emp_length` parsed to years; date fields parsed to datetime; derived `credit_length_years`.
- Mapped `loan_status` to `target_default` using domain rules: 'Fully Paid' -> 0, 'Charged Off'/'Default'/'Late (120)' -> 1. Rows without clearly-defined outcome (e.g., current loans) were dropped to avoid label noise.
- Dropped columns with >75% missingness (adjustable threshold) to reduce noise.
- Built a reproducible `preprocessor` pipeline (ColumnTransformer) that:
  - median-imputes numeric data, winsorizes (1–99 percentile), and standard-scales.
  - imputes categorical variables with 'missing' and one-hot encodes them.
- Produced stratified train/test split to preserve class balance.

Rationale:
This pipeline is conservative and reproduces transformations in test/inference. Winsorizing limits extreme outliers that skew models; imputation choices are simple but robust. Temporal split is recommended for deployment but stratified split was used for model development.

3. Models & training
--------------------
a) XGBoost (classical baseline)
- Hyperparameters: n_estimators=300, lr=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8.
- Training on preprocessed features.
- Metrics reported: ROC AUC, F1, precision/recall, and confusion matrix.
- Feature importance extracted (if preprocessor supports feature names).

b) Keras ANN (deep learning)
- Architecture: Dense(128)->Dropout->Dense(64)->Dropout->Dense(1, sigmoid)
- Loss: binary_crossentropy; optimizer: Adam; metrics: accuracy & AUC.
- Trained with 15 epochs and early validation split.

c) PyTorch MLP (alternate deep model)
- Architecture similar to Keras model; trained with BCEWithLogitsLoss and Adam.

4. Results (example)
--------------------
(Replace with your dataset-specific numbers after running the notebook.)

- XGBoost: AUC = 0.78, F1 = 0.45
- Keras ANN: Test AUC = 0.76, Test accuracy = 0.88
- PyTorch MLP: AUC = 0.75
- RL policy estimated value (avg reward per applicant): e.g., 120.5 (compare with always-approve and always-deny baselines)

Interpretation:
- AUC/F1 for classifiers capture ranking ability and balance of precision/recall. AUC is threshold-agnostic and useful for model selection; F1 summarizes performance at the chosen threshold and is sensitive to class imbalance.


5. Comparison and disagreement analysis
--------------------------------------
- DL classifier gives a score for default probability; a business rule (threshold) converts it to an approve/deny decision.


6. Limitations & future steps
-----------------------------
- Off-policy evaluation used is naive (direct averaging). For trustworthy deployment evaluate policies using IPS / Doubly Robust estimators or conduct randomized A/B testing.
- Reward engineering here is simplified: it ignores time value of money, recovery rates, and collections costs. A better reward should discount future payments, include expected recovery fractions, and account for operational costs.
- The dataset may have leakage if features include post-funding events; we printed potential leakage columns for review. For production, remove any feature that wouldn't be available at decision time.
- Consider temporal splitting (train on loans issued before year N, test on later loans) to simulate live deployment.


7. Reproducibility
------------------
- Preprocessing pipeline saved as `preprocessor.joblib`.
- Models saved as `xgb_model.json`, `keras_ann.h5`, and `pytorch_mlp.pt`.
- Use `requirements.txt` to recreate environment.

Appendix: Code pointers
- See `Loan_Project_Consolidated.ipynb` cells for exact code used for preprocessing, modeling, policy derivation, and analysis.
