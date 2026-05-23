import pandas as pd
import numpy as np


class QCADiagnostics:
    """
    Reliability diagnostics for the Text Classification to QCA Analysis Tool.

    This module provides three diagnostic functions:

    1. Prototype quality checks
       Check whether conceptual prototypes are complete, unique, and usable.

    2. Top-condition explanation table
       Show which condition each text is most strongly associated with.

    3. Threshold suggestion
       Suggest possible calibration thresholds based on the empirical
       distribution of prototype similarity scores.
    """

    def __init__(self):
        pass

    def check_prototype_quality(self, prototype_df):
        """
        Check prototype file quality.

        Parameters
        ----------
        prototype_df : pd.DataFrame
            Prototype dataframe with columns:
            - condition_name
            - prototype
            - type

        Returns
        -------
        pd.DataFrame
            Diagnostic table with issue type, severity, and message.
        """
        issues = []

        required_cols = ["condition_name", "prototype", "type"]

        for col in required_cols:
            if col not in prototype_df.columns:
                issues.append({
                    "check": "required_column",
                    "severity": "error",
                    "message": f"Missing required column: {col}"
                })

        if issues:
            return pd.DataFrame(issues)

        df = prototype_df.copy()

        df["condition_name"] = df["condition_name"].astype(str)
        df["prototype"] = df["prototype"].astype(str)
        df["type"] = df["type"].astype(str)

        # Empty values
        for col in required_cols:
            empty_rows = df[df[col].str.strip().isin(["", "nan", "None"])]
            if not empty_rows.empty:
                issues.append({
                    "check": "empty_value",
                    "severity": "error",
                    "message": f"Column '{col}' contains empty values in row(s): {empty_rows.index.tolist()}"
                })

        # Duplicate condition names
        duplicated = df[df["condition_name"].duplicated(keep=False)]
        if not duplicated.empty:
            duplicate_names = duplicated["condition_name"].unique().tolist()
            issues.append({
                "check": "duplicate_condition_name",
                "severity": "error",
                "message": f"Duplicated condition_name values found: {duplicate_names}"
            })

        # Type validity
        valid_types = {"condition", "outcome"}
        invalid_types = sorted(set(df["type"].unique()) - valid_types)
        if invalid_types:
            issues.append({
                "check": "invalid_type",
                "severity": "error",
                "message": f"Invalid type value(s): {invalid_types}. Allowed values are 'condition' and 'outcome'."
            })

        # At least one condition prototype
        n_conditions = (df["type"] == "condition").sum()
        if n_conditions == 0:
            issues.append({
                "check": "no_condition_prototype",
                "severity": "error",
                "message": "No condition prototype found. At least one row with type='condition' is required."
            })

        # Short prototypes
        df["prototype_word_count"] = df["prototype"].apply(lambda x: len(str(x).split()))
        short_rows = df[df["prototype_word_count"] < 5]
        if not short_rows.empty:
            issues.append({
                "check": "short_prototype",
                "severity": "warning",
                "message": (
                    "Some prototypes are very short. Short prototypes may produce unstable "
                    f"semantic similarity scores. Affected condition(s): "
                    f"{short_rows['condition_name'].tolist()}"
                )
            })

        # Very long prototypes
        long_rows = df[df["prototype_word_count"] > 80]
        if not long_rows.empty:
            issues.append({
                "check": "long_prototype",
                "severity": "warning",
                "message": (
                    "Some prototypes are very long. Very long prototypes may mix multiple concepts. "
                    f"Affected condition(s): {long_rows['condition_name'].tolist()}"
                )
            })

        # Outcome prototype notice
        n_outcomes = (df["type"] == "outcome").sum()
        if n_outcomes > 0:
            issues.append({
                "check": "outcome_prototype_notice",
                "severity": "info",
                "message": (
                    "Outcome prototype(s) detected. In the current workflow, condition prototypes "
                    "are used for text scoring, while the outcome variable is taken from the uploaded text dataset."
                )
            })

        if not issues:
            issues.append({
                "check": "prototype_quality",
                "severity": "pass",
                "message": "No major prototype quality issue detected."
            })

        return pd.DataFrame(issues)

    def create_top_condition_explanation(self, score_df):
        """
        Create a table explaining the strongest condition match for each case.

        Parameters
        ----------
        score_df : pd.DataFrame
            Score table generated by TextPrototypeScorer.

        Returns
        -------
        pd.DataFrame
            Explanation table containing top and second-best matched conditions.
        """
        score_cols = [col for col in score_df.columns if col.endswith("_score")]

        if not score_cols:
            raise ValueError("No score columns found. Expected columns ending with '_score'.")

        rows = []

        for _, row in score_df.iterrows():
            scores = row[score_cols].astype(float).sort_values(ascending=False)

            top_score_col = scores.index[0]
            top_score = scores.iloc[0]

            if len(scores) > 1:
                second_score_col = scores.index[1]
                second_score = scores.iloc[1]
            else:
                second_score_col = None
                second_score = np.nan

            top_condition = top_score_col.replace("_score", "")
            second_condition = (
                second_score_col.replace("_score", "")
                if second_score_col is not None else None
            )

            gap = top_score - second_score if not pd.isna(second_score) else np.nan

            result = {
                "case_id": row["case_id"] if "case_id" in score_df.columns else None,
                "text": row["text"] if "text" in score_df.columns else None,
                "top_condition": top_condition,
                "top_score": round(float(top_score), 4),
                "second_condition": second_condition,
                "second_score": round(float(second_score), 4) if not pd.isna(second_score) else np.nan,
                "score_gap": round(float(gap), 4) if not pd.isna(gap) else np.nan
            }

            if "outcome" in score_df.columns:
                result["outcome"] = row["outcome"]

            rows.append(result)

        explanation_df = pd.DataFrame(rows)

        return explanation_df

    def suggest_thresholds(self, score_df):
        """
        Suggest calibration thresholds from raw similarity score distribution.

        The suggested values are not automatically imposed. They are meant to
        inform the researcher's calibration decision.

        Parameters
        ----------
        score_df : pd.DataFrame
            Score table generated by TextPrototypeScorer.

        Returns
        -------
        pd.DataFrame
            Suggested thresholds and summary statistics.
        """
        score_cols = [col for col in score_df.columns if col.endswith("_score")]

        if not score_cols:
            raise ValueError("No score columns found. Expected columns ending with '_score'.")

        values = score_df[score_cols].values.flatten()
        values = pd.Series(values).dropna().astype(float)

        if values.empty:
            raise ValueError("No valid score values found.")

        threshold_table = pd.DataFrame([
            {
                "rule": "conservative",
                "suggested_threshold": round(values.quantile(0.75), 4),
                "interpretation": "Higher threshold; fewer cases are assigned to condition sets."
            },
            {
                "rule": "balanced",
                "suggested_threshold": round(values.quantile(0.60), 4),
                "interpretation": "Moderate threshold; useful as a starting point for crisp-set calibration."
            },
            {
                "rule": "inclusive",
                "suggested_threshold": round(values.quantile(0.50), 4),
                "interpretation": "Lower threshold; more cases are assigned to condition sets."
            }
        ])

        summary_table = pd.DataFrame([
            {
                "statistic": "minimum",
                "value": round(values.min(), 4)
            },
            {
                "statistic": "25th percentile",
                "value": round(values.quantile(0.25), 4)
            },
            {
                "statistic": "median",
                "value": round(values.median(), 4)
            },
            {
                "statistic": "60th percentile",
                "value": round(values.quantile(0.60), 4)
            },
            {
                "statistic": "75th percentile",
                "value": round(values.quantile(0.75), 4)
            },
            {
                "statistic": "maximum",
                "value": round(values.max(), 4)
            }
        ])

        return threshold_table, summary_table