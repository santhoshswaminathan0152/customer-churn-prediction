"""Train, tune and compare churn models; pick a business-driven threshold; explain with SHAP.

Usage: python -m src.train
"""
import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import ROOT, TARGET, load
from src.features import make_preprocessor

REPORTS, MODELS = ROOT / "reports", ROOT / "models"
SEED = 42

# Business assumption (illustrative, not from the dataset): a retention offer costs $50 per targeted
# customer, a lost customer costs $500 in future revenue, and the offer saves 50% of real churners.
OFFER_COST, CHURN_COST, OFFER_SUCCESS = 50, 500, 0.5


def build(name, pos_weight):
    feats, cols = make_preprocessor(scale=name == "logistic_regression")
    model, grid = {
        "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"model__C": np.logspace(-2, 1, 8)}),
        "random_forest": (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED),
                          {"model__n_estimators": [200, 400], "model__max_depth": [6, 8, 12, None],
                           "model__min_samples_leaf": [1, 5, 10, 20]}),
        "xgboost": (XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1, random_state=SEED),
                    {"model__n_estimators": [200, 300, 500], "model__max_depth": [2, 3, 4, 5],
                     "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0],
                     "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 5, 10]}),
    }[name]
    return Pipeline([("features", feats), ("prep", cols), ("model", model)]), grid


def business_cost(y, proba, t):
    """Total cost of a targeting policy: offer everyone above threshold t."""
    target = proba >= t
    tp, fp, fn = (target & (y == 1)).sum(), (target & (y == 0)).sum(), (~target & (y == 1)).sum()
    return (tp + fp) * OFFER_COST + tp * (1 - OFFER_SUCCESS) * CHURN_COST + fn * CHURN_COST


def scores(y, proba, t):
    pred = (proba >= t).astype(int)
    return {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "precision": precision_score(y, pred), "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load()
    X, y = df.drop(columns=TARGET), df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"{len(df)} customers, churn rate {y.mean():.1%}, train {len(X_train)}, test {len(X_test)}")

    rows, fitted, test_proba = [], {}, {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        pipe, grid = build(name, pos_weight)
        search = RandomizedSearchCV(pipe, grid, n_iter=min(25, np.prod([len(v) for v in grid.values()])),
                                    scoring="average_precision", cv=cv, n_jobs=-1, random_state=SEED)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        test_proba[name] = search.best_estimator_.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "cv_pr_auc": search.best_score_, **scores(y_test, test_proba[name], 0.5)})
        print(f"{name:20s} cv_pr_auc={search.best_score_:.3f} test_roc_auc={rows[-1]['roc_auc']:.3f} "
              f"test_pr_auc={rows[-1]['pr_auc']:.3f}")

    results = pd.DataFrame(rows)
    best_name = results.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]  # chosen on CV, not test
    best = fitted[best_name]

    # Threshold from out-of-fold predictions on the training set only, so the test set stays untouched.
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = np.array([business_cost(y_train, oof, t) for t in thresholds])
    threshold = float(thresholds[costs.argmin()])

    p = test_proba[best_name]
    tuned = scores(y_test, p, threshold)
    cost_none = business_cost(y_test, p, 1.01)
    cost_default = business_cost(y_test, p, 0.5)
    cost_tuned = business_cost(y_test, p, threshold)
    business = {"best_model": best_name, "threshold": threshold, "offer_cost": OFFER_COST, "churn_cost": CHURN_COST,
                "offer_success": OFFER_SUCCESS, "test_cost_no_action": int(cost_none),
                "test_cost_threshold_0.5": int(cost_default), "test_cost_tuned": int(cost_tuned),
                "test_savings_vs_no_action": int(cost_none - cost_tuned), **{f"tuned_{k}": v for k, v in tuned.items()}}
    results.round(4).to_csv(REPORTS / "metrics.csv", index=False)
    (REPORTS / "business_metrics.json").write_text(json.dumps(business, indent=2, default=float))
    print(json.dumps(business, indent=2, default=float))

    plot_curves(y_test, test_proba)
    plot_cost(thresholds, costs, threshold)
    plot_confusion(y_test, p, threshold, best_name)
    explain(best, X_train, X_test, p, best_name)
    joblib.dump({"pipeline": best, "threshold": threshold, "model_name": best_name}, MODELS / "churn_model.joblib")
    with open(REPORTS / "metrics_log.jsonl", "a") as log:
        log.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                              "results": results.round(4).to_dict("records"), "business": business}, default=float) + "\n")


def plot_curves(y, probas):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        b.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    a.plot([0, 1], [0, 1], "k--", alpha=.4)
    b.axhline(y.mean(), color="k", ls="--", alpha=.4, label=f"baseline ({y.mean():.2f})")
    a.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    b.set(title="Precision-Recall curve (test set)", xlabel="Recall", ylabel="Precision")
    for ax in (a, b):
        ax.legend(loc="lower right" if ax is a else "upper right")
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_pr_curves.png", dpi=120)
    plt.close()


def plot_cost(thresholds, costs, best):
    plt.figure(figsize=(8, 4.5))
    plt.plot(thresholds, costs / 1000, color="#7c3aed")
    plt.axvline(best, color="#16a34a", ls="--", label=f"chosen threshold = {best:.2f}")
    plt.axvline(0.5, color="#9ca3af", ls=":", label="default 0.5")
    plt.title("Business cost vs decision threshold (out-of-fold, training set)")
    plt.xlabel("threshold")
    plt.ylabel("total cost ($ thousands)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "cost_vs_threshold.png", dpi=120)
    plt.close()


def plot_confusion(y, p, t, name):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, (p >= t).astype(int), display_labels=["Stay", "Churn"],
                                            cmap="Purples", colorbar=False, ax=ax)
    ax.set_title(f"{name} @ threshold {t:.2f} (test set)")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close()


def explain(pipe, X_train, X_test, p, name):
    prep = Pipeline(pipe.steps[:-1])
    names = pipe.named_steps["prep"].get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X_test), columns=names)
    model = pipe.named_steps["model"]
    if name == "logistic_regression":
        explainer = shap.LinearExplainer(model, pd.DataFrame(prep.transform(X_train), columns=names))
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    if sv.values.ndim == 3:  # random forest returns one set per class
        sv = sv[:, :, 1]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: what drives churn predictions (test set)")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    i = int(np.argmax(p))
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"Why this customer is high-risk (churn probability {p[i]:.0%})")
    plt.savefig(REPORTS / "shap_customer.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
