
import pandas as pd
import numpy as np

from src.calibration import QCACalibrator
from src.qca import BasicQCAAnalyzer


class ThresholdSensitivityAnalyzer:
    """
    Threshold sensitivity analysis for crisp-set QCA.

    This module evaluates how QCA results change when the crisp-set calibration
    threshold changes. It is designed to work with the existing score table,
    calibration module, and QCA module.

    For each threshold, it reports:
    - number of observed configurations
    - number of sufficient configurations
    - number of contradictory configurations
    - number of weak configurations
    - average consistency
    - maximum consistency
    - average raw coverage
    - configuration-level results
    """

    def __init__(
        self,
        thresholds=None,
        consistency_cutoff=0.80,
        min_cases=1,
        contradiction_lower=0.40,
        contradiction_upper=0.60
    ):
        if thresholds is None:
            thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

        self.thresholds = thresholds
        self.consistency_cutoff = consistency_cutoff
        self.min_cases = min_cases
        self.contradiction_lower = contradiction_lower
        self.contradiction_upper = contradiction_upper

    def run(
        self,
        score_df,
        outcome_column="outcome",
        case_id_column="case_id"
    ):
        """
        Run threshold sensitivity analysis.

        Parameters
        ----------
        score_df : pd.DataFrame
            Score table containing *_score columns.
        outcome_column : str
            Outcome column name.
        case_id_column : str
            Case ID column name.

        Returns
        -------
        summary_df : pd.DataFrame
            One row per threshold.
        detail_df : pd.DataFrame
            One row per observed configuration per threshold.
        solution_df : pd.DataFrame
            One row per sufficient solution configuration per threshold.
        """
        calibrator = QCACalibrator()

        summary_rows = []
        detail_tables = []
        solution_tables = []

        for threshold in self.thresholds:
            crisp_df = calibrator.calibrate_dataframe(
                score_df=score_df,
                method="crisp",
                threshold=threshold,
                keep_score_columns=True
            )

            qca_ready_df = calibrator.create_qca_ready_dataset(
                calibrated_df=crisp_df,
                outcome_column=outcome_column,
                case_id_column=case_id_column
            )

            analyzer = BasicQCAAnalyzer(
                consistency_cutoff=self.consistency_cutoff,
                min_cases=self.min_cases,
                contradiction_lower=self.contradiction_lower,
                contradiction_upper=self.contradiction_upper
            )

            truth_table = analyzer.create_truth_table(
                df=qca_ready_df,
                case_id_column=case_id_column,
                outcome_column=outcome_column
            )

            solution_configs = analyzer.extract_solution_configurations(truth_table)

            truth_table = truth_table.copy()
            truth_table.insert(0, "threshold", threshold)

            if not solution_configs.empty:
                solution_configs = solution_configs.copy()
                solution_configs.insert(0, "threshold", threshold)

            detail_tables.append(truth_table)
            solution_tables.append(solution_configs)

            status_counts = truth_table["status"].value_counts().to_dict()

            summary_rows.append({
                "threshold": threshold,
                "n_configurations": int(len(truth_table)),
                "n_sufficient": int(status_counts.get("sufficient", 0)),
                "n_contradictory": int(status_counts.get("contradictory", 0)),
                "n_weak": int(status_counts.get("weak", 0)),
                "mean_consistency": round(float(truth_table["consistency"].mean()), 4),
                "max_consistency": round(float(truth_table["consistency"].max()), 4),
                "mean_raw_coverage": round(float(truth_table["raw_coverage"].mean()), 4),
                "max_raw_coverage": round(float(truth_table["raw_coverage"].max()), 4),
                "total_cases_in_truth_table": int(truth_table["n_cases"].sum())
            })

        summary_df = pd.DataFrame(summary_rows)

        detail_df = (
            pd.concat(detail_tables, ignore_index=True)
            if detail_tables else pd.DataFrame()
        )

        non_empty_solutions = [
            df for df in solution_tables
            if df is not None and not df.empty
        ]

        solution_df = (
            pd.concat(non_empty_solutions, ignore_index=True)
            if non_empty_solutions else pd.DataFrame()
        )

        return summary_df, detail_df, solution_df

    def create_configuration_stability_table(self, detail_df):
        """
        Create a configuration stability table across thresholds.

        A configuration is more stable if it appears as sufficient across
        multiple thresholds.
        """
        if detail_df.empty:
            return pd.DataFrame()

        required_cols = ["threshold", "configuration", "status", "consistency", "raw_coverage", "n_cases"]
        for col in required_cols:
            if col not in detail_df.columns:
                raise ValueError(f"Missing required column: {col}")

        rows = []

        for configuration, group in detail_df.groupby("configuration"):
            thresholds_observed = sorted(group["threshold"].unique().tolist())
            sufficient_thresholds = sorted(
                group.loc[group["status"] == "sufficient", "threshold"].unique().tolist()
            )
            contradictory_thresholds = sorted(
                group.loc[group["status"] == "contradictory", "threshold"].unique().tolist()
            )
            weak_thresholds = sorted(
                group.loc[group["status"] == "weak", "threshold"].unique().tolist()
            )

            rows.append({
                "configuration": configuration,
                "n_thresholds_observed": len(thresholds_observed),
                "n_thresholds_sufficient": len(sufficient_thresholds),
                "n_thresholds_contradictory": len(contradictory_thresholds),
                "n_thresholds_weak": len(weak_thresholds),
                "thresholds_observed": ", ".join([str(x) for x in thresholds_observed]),
                "sufficient_thresholds": ", ".join([str(x) for x in sufficient_thresholds]),
                "contradictory_thresholds": ", ".join([str(x) for x in contradictory_thresholds]),
                "weak_thresholds": ", ".join([str(x) for x in weak_thresholds]),
                "mean_consistency": round(float(group["consistency"].mean()), 4),
                "mean_raw_coverage": round(float(group["raw_coverage"].mean()), 4),
                "mean_n_cases": round(float(group["n_cases"].mean()), 4)
            })

        stability_df = pd.DataFrame(rows)

        if not stability_df.empty:
            stability_df = stability_df.sort_values(
                by=["n_thresholds_sufficient", "mean_consistency", "mean_raw_coverage"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

        return stability_df
