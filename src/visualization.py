import os
import pandas as pd
import plotly.express as px


class QCAVisualizer:
    """
    Visualization module for the Text Classification to QCA Analysis Tool.

    This module generates publication-oriented and reviewer-friendly figures:
    1. Raw prototype similarity score distribution
    2. Fuzzy-set membership heatmap
    3. Crisp-set membership heatmap
    4. Consistency-coverage plot for QCA configurations

    Each figure is exported as an interactive HTML file. If kaleido is installed,
    each figure is also exported as a static PNG file.
    """

    def __init__(self, export_png=True):
        """
        Parameters
        ----------
        export_png : bool
            Whether to export static PNG figures in addition to HTML figures.
            PNG export requires the kaleido package.
        """
        self.export_png = export_png

    def load_data(self, file_path):
        """
        Load a CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return pd.read_csv(file_path)

    def save_figure(self, fig, html_path):
        """
        Save Plotly figure as HTML and optionally PNG.
        """
        output_dir = os.path.dirname(html_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fig.write_html(html_path)
        print(f"Saved HTML figure to: {html_path}")

        if self.export_png:
            png_path = html_path.replace(".html", ".png")
            try:
                fig.write_image(png_path, scale=2)
                print(f"Saved PNG figure to: {png_path}")
            except Exception as e:
                print(
                    "PNG export skipped. To enable PNG export, install kaleido with: "
                    "pip install kaleido"
                )
                print(f"Reason: {e}")

    def create_score_distribution_plot(
        self,
        score_df,
        output_path="outputs/figure_1_score_distribution.html",
        crisp_threshold=0.30
    ):
        """
        Create a raw similarity score distribution plot.

        This figure helps justify the calibration threshold by showing how
        prototype similarity scores are distributed across QCA conditions.
        """
        score_cols = [col for col in score_df.columns if col.endswith("_score")]

        if not score_cols:
            raise ValueError("No score columns found. Expected columns ending with '_score'.")

        id_vars = [col for col in ["case_id", "outcome"] if col in score_df.columns]

        long_df = score_df.melt(
            id_vars=id_vars,
            value_vars=score_cols,
            var_name="condition",
            value_name="similarity_score"
        )

        long_df["condition"] = long_df["condition"].str.replace("_score", "", regex=False)

        fig = px.box(
            long_df,
            x="condition",
            y="similarity_score",
            points="all",
            hover_data=id_vars,
            title="Figure 1. Distribution of Prototype Similarity Scores",
            labels={
                "condition": "QCA condition",
                "similarity_score": "Raw prototype similarity score",
                "case_id": "Case ID",
                "outcome": "Outcome"
            }
        )

        fig.add_hline(
            y=crisp_threshold,
            line_dash="dash",
            annotation_text=f"Crisp-set threshold = {crisp_threshold:.2f}",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title_x=0.5,
            width=950,
            height=580,
            xaxis_title="Condition",
            yaxis_title="Raw similarity score",
            margin=dict(l=60, r=40, t=80, b=100),
            font=dict(size=13)
        )

        fig.add_annotation(
            text=(
                "Note: Points represent case-level similarity scores. "
                "The dashed line indicates the threshold used for crisp-set calibration."
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.22,
            showarrow=False,
            align="left",
            font=dict(size=11)
        )

        self.save_figure(fig, output_path)
        return fig

    def create_membership_heatmap(
        self,
        membership_df,
        output_path,
        title,
        case_id_column="case_id",
        outcome_column="outcome"
    ):
        """
        Create a heatmap of calibrated QCA membership values.
        """
        exclude_cols = [case_id_column, "text", outcome_column]
        score_cols = [col for col in membership_df.columns if col.endswith("_score")]

        condition_cols = [
            col for col in membership_df.columns
            if col not in exclude_cols and col not in score_cols
        ]

        if not condition_cols:
            raise ValueError("No condition columns found for heatmap.")

        heatmap_df = membership_df[[case_id_column] + condition_cols].copy()
        heatmap_df[case_id_column] = heatmap_df[case_id_column].astype(str)

        heatmap_matrix = heatmap_df.set_index(case_id_column)[condition_cols]

        fig = px.imshow(
            heatmap_matrix,
            text_auto=".2f",
            aspect="auto",
            labels=dict(
                x="QCA condition",
                y="Case ID",
                color="Membership"
            ),
            title=title,
            zmin=0,
            zmax=1
        )

        fig.update_layout(
            title_x=0.5,
            width=950,
            height=580,
            xaxis_title="Condition",
            yaxis_title="Case ID",
            margin=dict(l=70, r=40, t=80, b=100),
            font=dict(size=13)
        )

        fig.add_annotation(
            text=(
                "Note: Each cell reports the calibrated set-membership value "
                "of a case in a given QCA condition."
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.22,
            showarrow=False,
            align="left",
            font=dict(size=11)
        )

        self.save_figure(fig, output_path)
        return fig

    def create_consistency_coverage_plot(
        self,
        truth_table_df,
        output_path="outputs/figure_4_consistency_coverage_plot.html",
        consistency_cutoff=0.80
    ):
        """
        Create a consistency-coverage scatter plot.

        This figure helps identify sufficient, weak, and contradictory
        configurations in the QCA truth table.
        """
        required_cols = [
            "configuration",
            "consistency",
            "raw_coverage",
            "n_cases",
            "status",
            "cases"
        ]

        for col in required_cols:
            if col not in truth_table_df.columns:
                raise ValueError(f"Missing required column in truth table: {col}")

        fig = px.scatter(
            truth_table_df,
            x="raw_coverage",
            y="consistency",
            size="n_cases",
            color="status",
            hover_name="configuration",
            hover_data={
                "n_cases": True,
                "raw_coverage": ":.4f",
                "consistency": ":.4f",
                "cases": True,
                "status": True
            },
            title="Figure 4. Consistency-Coverage Plot of QCA Configurations",
            labels={
                "raw_coverage": "Raw coverage",
                "consistency": "Consistency",
                "n_cases": "Number of cases",
                "status": "Configuration status",
                "cases": "Case IDs"
            }
        )

        fig.add_hline(
            y=consistency_cutoff,
            line_dash="dash",
            annotation_text=f"Consistency cutoff = {consistency_cutoff:.2f}",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title_x=0.5,
            width=950,
            height=580,
            xaxis=dict(range=[-0.02, 1.02], title="Raw coverage"),
            yaxis=dict(range=[-0.02, 1.05], title="Consistency"),
            margin=dict(l=70, r=40, t=80, b=100),
            font=dict(size=13)
        )

        fig.add_annotation(
            text=(
                "Note: Each point represents one observed configuration. "
                "Configurations above the dashed line meet the consistency threshold."
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.22,
            showarrow=False,
            align="left",
            font=dict(size=11)
        )

        self.save_figure(fig, output_path)
        return fig


def run_demo():
    """
    Run demo visualization workflow.
    """
    score_file = "outputs/score_table.csv"
    fuzzy_membership_file = "outputs/calibrated_membership_table_fuzzy.csv"
    crisp_membership_file = "outputs/calibrated_membership_table_crisp.csv"
    truth_table_file = "outputs/truth_table.csv"

    visualizer = QCAVisualizer(export_png=True)

    score_df = visualizer.load_data(score_file)
    fuzzy_df = visualizer.load_data(fuzzy_membership_file)
    crisp_df = visualizer.load_data(crisp_membership_file)
    truth_table_df = visualizer.load_data(truth_table_file)

    print("\nCreating Figure 1: score distribution plot...")
    visualizer.create_score_distribution_plot(
        score_df=score_df,
        output_path="outputs/figure_1_score_distribution.html",
        crisp_threshold=0.30
    )

    print("\nCreating Figure 2: fuzzy-set membership heatmap...")
    visualizer.create_membership_heatmap(
        membership_df=fuzzy_df,
        output_path="outputs/figure_2_fuzzy_membership_heatmap.html",
        title="Figure 2. Fuzzy-Set Membership Heatmap",
        case_id_column="case_id",
        outcome_column="outcome"
    )

    print("\nCreating Figure 3: crisp-set membership heatmap...")
    visualizer.create_membership_heatmap(
        membership_df=crisp_df,
        output_path="outputs/figure_3_crisp_membership_heatmap.html",
        title="Figure 3. Crisp-Set Membership Heatmap",
        case_id_column="case_id",
        outcome_column="outcome"
    )

    print("\nCreating Figure 4: consistency-coverage plot...")
    visualizer.create_consistency_coverage_plot(
        truth_table_df=truth_table_df,
        output_path="outputs/figure_4_consistency_coverage_plot.html",
        consistency_cutoff=0.80
    )

    print("\nOptimized visualization step completed.")


if __name__ == "__main__":
    run_demo()