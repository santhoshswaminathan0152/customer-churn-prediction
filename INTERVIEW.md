# Interview Questions: Customer Churn Prediction

Likely questions about this project, with answers based on the actual code and results.

---

### Problem & data

**1. Walk me through the project in 60 seconds.**
I built an end-to-end churn model on the IBM Telco dataset (7,043 customers, 26.5% churn). I cleaned the data, did EDA, engineered features and compared Logistic Regression, Random Forest and XGBoost inside sklearn pipelines with 5-fold cross-validated tuning. XGBoost was best on CV PR-AUC (0.666) and scored 0.847 ROC-AUC / 0.663 PR-AUC on the test set. Instead of using 0.5 blindly, I chose the threshold with a business cost model, explained predictions with SHAP and served the model through a FastAPI endpoint.

**2. What data quality issues did you find?**
`TotalCharges` was stored as text, and 11 rows were blank. Every one of them had `tenure = 0`, so these are brand-new customers who haven't been billed yet. I filled them with 0, which is correct for that business case; mean imputation would have invented charges. I also deduplicate on `customerID` only, because two different customers can legitimately have identical attributes.

**3. What were the strongest churn signals in EDA?**
Contract type: 42.7% churn on month-to-month vs 2.8% on two-year contracts. Tenure: 52.9% churn in the first 6 months vs 9.5% after 4+ years. Electronic check payers churn at 45.3%, fiber optic customers at 41.9% (vs 19.0% on DSL), and customers without tech support at 41.6% (vs 15.2% with it).

**4. Which features did you engineer and why?**
Tenure buckets (churn is very non-linear in tenure), number of services (more products → more lock-in), average monthly spend (`TotalCharges / tenure`), `charge_increase` (current bill minus historical average, a proxy for a recent price rise), a family flag (partner or dependents) and an auto-payment flag. They are computed inside the pipeline by a `FunctionTransformer`, so training and the API use exactly the same logic.

---

### Modelling

**5. Why PR-AUC instead of accuracy?**
With 26.5% churners, a model that predicts "stay" for everyone gets ~73.5% accuracy and is useless. PR-AUC focuses on the positive (churn) class and how precision trades off against recall; its random baseline is the churn rate (0.265), while ours is 0.663. ROC-AUC is also reported, but it can look optimistic on imbalanced data because true negatives dominate the false positive rate.

**6. How did you handle class imbalance?**
`class_weight="balanced"` for Logistic Regression and Random Forest, `scale_pos_weight` (≈2.77, the negative/positive ratio) for XGBoost, stratified train/test split and stratified K-fold. I didn't use SMOTE: re-weighting gives a similar effect without creating synthetic customers, and it's simpler to put in a pipeline.

**7. How did you prevent data leakage?**
- Every preprocessing step (feature engineering, scaling, one-hot encoding) sits inside a `Pipeline`, so cross-validation fits them only on the training folds.
- The test set (20%, stratified) is used once, for the final evaluation.
- The best model is chosen on **cross-validated** PR-AUC, not on test scores.
- The decision threshold is tuned on **out-of-fold** predictions from the training set, not on the test set.

**8. Why did all three models perform almost the same?**
ROC-AUC is 0.845 / 0.843 / 0.847. Most of the signal comes from a few strong, mostly monotonic features (contract, tenure, services), which a linear model captures well. That's a useful finding in itself: logistic regression is nearly as good, cheaper and easier to explain to the business. I'd pick XGBoost only if the small lift is worth the extra complexity.

**9. How did you tune hyperparameters?**
`RandomizedSearchCV` with 5-fold stratified CV, optimising `average_precision` (PR-AUC). The best XGBoost settings were shallow trees (`max_depth=3`), a low learning rate (0.01) with 500 trees, `colsample_bytree=0.6` and `min_child_weight=5`: a strongly regularised model, consistent with a small, fairly simple dataset.

**10. Why scale features for logistic regression but not for the tree models?**
Logistic regression with L2 regularisation is sensitive to feature scale: unscaled features are penalised unequally and the solver converges more slowly. Trees split on thresholds, so scaling doesn't change them.

---

### Threshold & business impact

**11. How did you choose the decision threshold?**
With a cost model: a retention offer costs $50, a lost customer costs $500, and the offer saves 50% of churners. *These are assumptions I state clearly; in a real company they would come from finance and marketing.* I computed the total cost for thresholds 0.05–0.95 on out-of-fold training predictions and picked the minimum: **0.42**.

**12. What was the business result?**
On the 1,409 test customers, doing nothing costs $187,000. Targeting customers with p ≥ 0.42 costs $141,500, which saves **$45,500 (24%)**, and it catches 314 of 374 churners (84% recall) at 47.6% precision.

**13. Did the tuned threshold beat 0.5?**
Not really, and I'd say so openly. On the test set, 0.5 cost $141,250 and 0.42 cost $141,500. The cost curve is flat around its minimum, and because class weighting already shifts the probabilities towards churn, the optimum lands close to 0.5. The value of the exercise is the framework: if the offer cost or customer value changes, the threshold moves accordingly.

**14. Are the predicted probabilities calibrated?**
No. Class weighting inflates churn probabilities, so 0.42 should not be read as "42% chance". Before using probabilities directly (e.g. for expected-revenue forecasts) I would calibrate them with isotonic regression or Platt scaling via `CalibratedClassifierCV`. That's on the roadmap.

---

### Explainability

**15. How does SHAP work and what did it show?**
SHAP assigns each feature a contribution to a single prediction, based on Shapley values from game theory; the contributions add up to the model output (in log-odds for XGBoost). Globally, month-to-month contracts, short tenure, no online security, no tech support, fiber optic and electronic check increase churn risk, while two-year contracts strongly reduce it.

**16. How would a business user use the local explanation?**
For the highest-risk test customer (95%), SHAP shows the risk comes from 1-month tenure, a month-to-month contract and no online security or tech support. That tells the retention team *which* offer to make: a discounted annual contract or free security/tech support, rather than a generic discount.

**17. Is SHAP showing causation?**
No. It explains what the *model* relies on, not what *causes* churn. For example, fiber optic customers churn more, but that might be due to price or competition rather than the fiber itself. To measure the causal effect of an offer you need an A/B test or uplift modelling.

---

### Engineering & deployment

**18. How is the model served?**
A FastAPI app (`api.py`) loads the saved pipeline and threshold. `/predict` takes a customer JSON validated by a Pydantic schema (types, allowed categories, non-negative charges) and returns the probability, a churn flag and a risk level. Invalid input returns HTTP 422. `/health` reports the model in use.

**19. How did you test it?**
pytest: unit tests for the cleaning logic (blank `TotalCharges`, duplicate IDs, target encoding) and the engineered features, plus API tests for health, a high-risk customer, a loyal low-risk customer and invalid input. All 6 pass.

**20. How would you monitor this in production?**
Track input drift (e.g. Population Stability Index on tenure, charges and contract mix), the prediction distribution, and the real churn of scored customers once labels arrive (with a lag of about a month). I'd retrain on a schedule or when PR-AUC drops below an agreed level, and log model versions.

---

### Next steps

**21. What would you do with more time?**
1. **Uplift modelling:** target customers whose behaviour the offer can *change*, not just those likely to churn.
2. **Survival analysis** (e.g. Cox model) to predict *when* customers churn.
3. **Probability calibration** before the cost model.
4. Richer features: support tickets, usage trends, complaint history, price changes over time.
5. An A/B test of the retention campaign to measure real savings instead of assumed ones.

**22. What is the biggest limitation of this project?**
The business numbers ($50 offer, $500 customer value, 50% success) are assumptions, and the dataset is a single static snapshot with no time dimension, so I can't validate on future data (out-of-time validation). In a real project the first thing I'd do is get real cost numbers and a time-based split.
