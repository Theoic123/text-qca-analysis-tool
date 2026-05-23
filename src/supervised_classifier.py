
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SupervisedTextConditionClassifier:
    """
    Train one binary TF-IDF + Logistic Regression classifier per QCA condition.

    Expected labeled data format:
    case_id,text,dissatisfaction,policy_demand,coproduction_request
    1,居民对停车收费不透明非常不满。,1,0,0
    2,请说明申请补贴需要哪些材料。,0,1,0
    3,我们愿意参加社区志愿服务。,0,0,1
    """

    def __init__(
        self,
        text_column="text",
        condition_columns=None,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        random_state=42
    ):
        self.text_column = text_column
        self.condition_columns = condition_columns
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

        # char_wb works well for Chinese short text and does not require word segmentation.
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            analyzer="char_wb"
        )

        self.models = {}
        self.training_report_ = None
        self.validation_report_ = None
        self.is_fitted = False

    def detect_condition_columns(self, labeled_df, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = ["case_id", self.text_column, "outcome", "target_label", "label"]

        candidates = []
        for col in labeled_df.columns:
            if col in exclude_columns:
                continue

            values = set(labeled_df[col].dropna().unique().tolist())
            allowed = {0, 1, 0.0, 1.0, "0", "1"}
            if values.issubset(allowed):
                candidates.append(col)

        return candidates

    def validate_labeled_data(self, labeled_df):
        issues = []

        if self.text_column not in labeled_df.columns:
            issues.append({
                "severity": "error",
                "message": f"Text column '{self.text_column}' not found in labeled data."
            })
            return pd.DataFrame(issues)

        if self.condition_columns is None:
            self.condition_columns = self.detect_condition_columns(labeled_df)

        if not self.condition_columns:
            issues.append({
                "severity": "error",
                "message": (
                    "No binary condition label columns found. "
                    "Please provide manually labeled 0/1 columns for each QCA condition."
                )
            })
            return pd.DataFrame(issues)

        for col in self.condition_columns:
            if col not in labeled_df.columns:
                issues.append({
                    "severity": "error",
                    "message": f"Condition label column '{col}' not found."
                })
                continue

            numeric_y = pd.to_numeric(labeled_df[col], errors="coerce")
            raw_values = set(labeled_df[col].dropna().unique().tolist())
            allowed = {0, 1, 0.0, 1.0, "0", "1"}

            if not raw_values.issubset(allowed):
                issues.append({
                    "severity": "error",
                    "message": f"Column '{col}' must contain only 0/1 labels. Found: {sorted(raw_values)}"
                })
                continue

            y = numeric_y.fillna(0).astype(int)
            n_total = int(len(y))
            n_positive = int(y.sum())
            n_negative = int((y == 0).sum())

            if n_positive == 0:
                issues.append({
                    "severity": "error",
                    "message": f"Condition '{col}' has no positive examples. At least one label=1 is required."
                })
            if n_negative == 0:
                issues.append({
                    "severity": "error",
                    "message": f"Condition '{col}' has no negative examples. At least one label=0 is required."
                })
            if 0 < n_positive < 3 or 0 < n_negative < 3:
                issues.append({
                    "severity": "warning",
                    "message": (
                        f"Condition '{col}' has limited examples "
                        f"(positive={n_positive}, negative={n_negative}). Predictions may be unstable."
                    )
                })

        empty_text_count = labeled_df[self.text_column].astype(str).str.strip().isin(["", "nan", "None"]).sum()
        if empty_text_count > 0:
            issues.append({
                "severity": "warning",
                "message": f"{empty_text_count} row(s) have empty text values."
            })

        if not issues:
            issues.append({
                "severity": "pass",
                "message": "Labeled data passed basic validation checks."
            })

        self.validation_report_ = pd.DataFrame(issues)
        return self.validation_report_

    def fit(self, labeled_df):
        validation_df = self.validate_labeled_data(labeled_df)

        if "severity" in validation_df.columns and (validation_df["severity"] == "error").any():
            raise ValueError("Labeled data contains error-level issues. Please inspect the validation table.")

        x_text = labeled_df[self.text_column].astype(str).fillna("")
        X = self.vectorizer.fit_transform(x_text)

        report_rows = []

        for condition in self.condition_columns:
            y = pd.to_numeric(labeled_df[condition], errors="coerce").fillna(0).astype(int)

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
            model.fit(X, y)
            self.models[condition] = model

            n_positive = int(y.sum())
            n_negative = int((y == 0).sum())

            report_rows.append({
                "condition": condition,
                "n_training_cases": int(len(y)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_rate": round(n_positive / len(y), 4)
            })

        self.training_report_ = pd.DataFrame(report_rows)
        self.is_fitted = True
        return self

    def predict_scores(self, text_df):
        if not self.is_fitted:
            raise RuntimeError("The supervised classifier has not been fitted yet.")

        if self.text_column not in text_df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in prediction data.")

        x_text = text_df[self.text_column].astype(str).fillna("")
        X = self.vectorizer.transform(x_text)

        score_df = text_df.copy()

        for condition, model in self.models.items():
            score_df[f"{condition}_score"] = model.predict_proba(X)[:, 1]

        return score_df

    def fit_predict_scores(self, labeled_df, text_df):
        self.fit(labeled_df)
        return self.predict_scores(text_df)
