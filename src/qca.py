import os
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

class BasicQCAAnalyzer:
    """
    Basic crisp-set QCA analyzer.

    This module takes a QCA-ready dataset and produces:
    1. Truth table
    2. Consistency and coverage measures
    3. Simple solution configurations
    4. Labels for strong, weak, and contradictory configurations

    The current implementation is designed for transparent educational and research-tool
    demonstration purposes. It focuses on interpretability rather than advanced Boolean
    minimization.
    """

    def __init__(
        self,
        consistency_cutoff=0.80,
        min_cases=1,
        contradiction_lower=0.40,
        contradiction_upper=0.60
    ):
        """
        Parameters
        ----------
        consistency_cutoff : float
            Minimum consistency required for a configuration to be considered sufficient.
        min_cases : int
            Minimum number of cases required for a configuration to be retained.
        contradiction_lower : float
            Lower bound for identifying contradictory configurations.
        contradiction_upper : float
            Upper bound for identifying contradictory configurations.
        """
        self.consistency_cutoff = consistency_cutoff
        self.min_cases = min_cases
        self.contradiction_lower = contradiction_lower
        self.contradiction_upper = contradiction_upper

    def load_qca_ready_dataset(self, file_path):
        """
        Load QCA-ready dataset.

        Expected structure:
        - one case_id column
        - several condition columns
        - one outcome column

        Parameters
        ----------
        file_path : str
            Path to qca_ready_dataset.csv.

        Returns
        -------
        pd.DataFrame
            Loaded QCA-ready dataset.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"QCA-ready dataset not found: {file_path}")

        df = pd.read_csv(file_path)

        if "outcome" not in df.columns:
            raise ValueError("The dataset must contain an 'outcome' column.")

        return df

    def detect_condition_columns(
        self,
        df,
        case_id_column="case_id",
        outcome_column="outcome"
    ):
        """
        Detect QCA condition columns.

        Parameters
        ----------
        df : pd.DataFrame
            QCA-ready dataset.
        case_id_column : str
            Case ID column.
        outcome_column : str
            Outcome column.

        Returns
        -------
        list
            List of condition column names.
        """
        condition_cols = [
            col for col in df.columns
            if col not in [case_id_column, outcome_column]
        ]

        if not condition_cols:
            raise ValueError("No condition columns found.")

        return condition_cols

    def validate_crisp_set_data(
        self,
        df,
        condition_cols,
        outcome_column="outcome"
    ):
        """
        Validate whether all conditions and outcome are crisp-set values.

        Crisp-set QCA requires values to be 0 or 1.

        Parameters
        ----------
        df : pd.DataFrame
            QCA-ready dataset.
        condition_cols : list
            Condition columns.
        outcome_column : str
            Outcome column.
        """
        check_cols = condition_cols + [outcome_column]

        for col in check_cols:
            unique_values = sorted(df[col].dropna().unique().tolist())
            allowed_values = {0, 1, 0.0, 1.0}

            if not set(unique_values).issubset(allowed_values):
                raise ValueError(
                    f"Column '{col}' contains non-crisp values: {unique_values}. "
                    f"Please use crisp-set calibrated data for this basic QCA module."
                )

    def build_configuration_label(self, row, condition_cols):
        """
        Build readable configuration label.

        Example:
        dissatisfaction=1, policy_demand=0, coproduction_request=1
        becomes:
        dissatisfaction * ~policy_demand * coproduction_request

        Parameters
        ----------
        row : pd.Series
            A row containing condition values.
        condition_cols : list
            Condition column names.

        Returns
        -------
        str
            Configuration label.
        """
        parts = []

        for col in condition_cols:
            value = int(row[col])
            if value == 1:
                parts.append(col)
            else:
                parts.append(f"~{col}")

        return " * ".join(parts)

    def create_truth_table(
        self,
        df,
        case_id_column="case_id",
        outcome_column="outcome"
    ):
        """
        Create truth table with consistency and coverage.

        For crisp-set QCA:
        - consistency for sufficiency = number of outcome-positive cases in configuration
          divided by total cases in configuration.
        - raw coverage = number of outcome-positive cases in configuration
          divided by total outcome-positive cases in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            QCA-ready dataset.
        case_id_column : str
            Case ID column.
        outcome_column : str
            Outcome column.

        Returns
        -------
        pd.DataFrame
            Truth table.
        """
        condition_cols = self.detect_condition_columns(
            df,
            case_id_column=case_id_column,
            outcome_column=outcome_column
        )

        self.validate_crisp_set_data(
            df,
            condition_cols=condition_cols,
            outcome_column=outcome_column
        )

        total_positive_outcomes = df[outcome_column].sum()

        grouped = df.groupby(condition_cols, dropna=False)

        rows = []

        for config_values, group in grouped:
            if len(condition_cols) == 1:
                config_values = (config_values,)

            config_dict = {
                condition_cols[i]: int(config_values[i])
                for i in range(len(condition_cols))
            }

            n_cases = len(group)
            outcome_sum = int(group[outcome_column].sum())
            consistency = outcome_sum / n_cases if n_cases > 0 else np.nan
            raw_coverage = (
                outcome_sum / total_positive_outcomes
                if total_positive_outcomes > 0 else np.nan
            )

            case_ids = (
                group[case_id_column].astype(str).tolist()
                if case_id_column in group.columns
                else []
            )

            config_series = pd.Series(config_dict)
            configuration_label = self.build_configuration_label(
                config_series,
                condition_cols
            )

            if consistency >= self.consistency_cutoff and n_cases >= self.min_cases:
                configuration_status = "sufficient"
            elif self.contradiction_lower <= consistency <= self.contradiction_upper:
                configuration_status = "contradictory"
            else:
                configuration_status = "weak"

            row = {
                **config_dict,
                "configuration": configuration_label,
                "n_cases": n_cases,
                "outcome_positive_cases": outcome_sum,
                "consistency": round(consistency, 4),
                "raw_coverage": round(raw_coverage, 4),
                "cases": ", ".join(case_ids),
                "status": configuration_status
            }

            rows.append(row)

        truth_table = pd.DataFrame(rows)

        truth_table = truth_table.sort_values(
            by=["status", "consistency", "n_cases"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        return truth_table

    def extract_solution_configurations(self, truth_table):
        """
        Extract solution configurations from the truth table.

        This basic version selects configurations that meet:
        - consistency >= consistency_cutoff
        - n_cases >= min_cases
        - status == sufficient

        Parameters
        ----------
        truth_table : pd.DataFrame
            Truth table.

        Returns
        -------
        pd.DataFrame
            Solution configurations.
        """
        solution_df = truth_table[
            (truth_table["consistency"] >= self.consistency_cutoff)
            & (truth_table["n_cases"] >= self.min_cases)
            & (truth_table["status"] == "sufficient")
        ].copy()

        if solution_df.empty:
            solution_df = pd.DataFrame(
                columns=[
                    "solution_id",
                    "configuration",
                    "n_cases",
                    "consistency",
                    "raw_coverage",
                    "cases",
                    "interpretation"
                ]
            )
            return solution_df

        solution_df = solution_df.reset_index(drop=True)
        solution_df.insert(0, "solution_id", range(1, len(solution_df) + 1))

        solution_df["interpretation"] = solution_df.apply(
            lambda row: (
                f"When the configuration [{row['configuration']}] is present, "
                f"the outcome occurs in {row['outcome_positive_cases']} out of "
                f"{row['n_cases']} observed case(s)."
            ),
            axis=1
        )

        keep_cols = [
            "solution_id",
            "configuration",
            "n_cases",
            "outcome_positive_cases",
            "consistency",
            "raw_coverage",
            "cases",
            "interpretation"
        ]

        return solution_df[keep_cols]

    def save_dataframe(self, df, output_path):
        """
        Save dataframe to CSV.
        """
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved to: {output_path}")


def run_demo():
    """
    Run demo QCA workflow.
    """
    input_file = "outputs/qca_ready_dataset.csv"
    truth_table_output = "outputs/truth_table.csv"
    solution_output = "outputs/solution_configurations.csv"

    analyzer = BasicQCAAnalyzer(
        consistency_cutoff=0.80,
        min_cases=1,
        contradiction_lower=0.40,
        contradiction_upper=0.60
    )

    qca_df = analyzer.load_qca_ready_dataset(input_file)

    print("\nQCA-ready dataset preview:")
    print(qca_df.head())

    truth_table = analyzer.create_truth_table(
        df=qca_df,
        case_id_column="case_id",
        outcome_column="outcome"
    )

    print("\nTruth table:")
    print(truth_table)

    analyzer.save_dataframe(truth_table, truth_table_output)

    solution_df = analyzer.extract_solution_configurations(truth_table)

    print("\nSolution configurations:")
    print(solution_df)

    analyzer.save_dataframe(solution_df, solution_output)


if __name__ == "__main__":
    run_demo()