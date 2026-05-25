import os
import io
import zipfile
import pandas as pd
import streamlit as st
import plotly.express as px

from src.text_scoring import TextPrototypeScorer
from src.calibration import QCACalibrator
from src.qca import BasicQCAAnalyzer
from src.visualization import QCAVisualizer
from src.diagnostics import QCADiagnostics
from src.supervised_classifier import SupervisedTextConditionClassifier
from src.prototype_explainer import RichTextPrototypeExplainer

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Text Classification to QCA Analysis Tool",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# Helper functions
# ============================================================

@st.cache_resource
def load_scorer(model_name):
    """
    Cache the Sentence-BERT scorer so the model is not reloaded every time.
    """
    return TextPrototypeScorer(model_name=model_name)


def convert_df_to_csv_bytes(df):
    """
    Convert dataframe to CSV bytes for Streamlit download buttons.
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def create_zip_from_outputs(output_dict):
    """
    Create a zip file in memory from multiple dataframe outputs.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in output_dict.items():
            if isinstance(content, pd.DataFrame):
                csv_bytes = convert_df_to_csv_bytes(content)
                zip_file.writestr(filename, csv_bytes)
            elif isinstance(content, bytes):
                zip_file.writestr(filename, content)
            elif isinstance(content, str):
                zip_file.writestr(filename, content.encode("utf-8"))
            else:
                zip_file.writestr(filename, str(content).encode("utf-8"))

    zip_buffer.seek(0)
    return zip_buffer


def create_score_distribution_plot(score_df, crisp_threshold):
    """
    Create Plotly figure for raw similarity score distribution.
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
        hover_data=[col for col in ["case_id", "outcome"] if col in long_df.columns],
        title="Figure 1. Distribution of Prototype Similarity Scores",
        labels={
            "condition": "QCA condition",
            "similarity_score": "Raw prototype similarity score"
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
        height=520
    )

    return fig


def create_membership_heatmap(
    membership_df,
    title,
    case_id_column="case_id",
    outcome_column="outcome"
):
    """
    Create Plotly heatmap for calibrated membership values.
    """
    exclude_cols = [case_id_column, "text", outcome_column]
    score_cols = [col for col in membership_df.columns if col.endswith("_score")]

    condition_cols = [
        col for col in membership_df.columns
        if col not in exclude_cols and col not in score_cols
    ]

    if not condition_cols:
        raise ValueError("No condition columns found for the heatmap.")

    if case_id_column not in membership_df.columns:
        raise ValueError(f"Case ID column '{case_id_column}' not found.")

    heatmap_df = membership_df[[case_id_column] + condition_cols].copy()
    heatmap_df[case_id_column] = heatmap_df[case_id_column].astype(str)

    heatmap_matrix = heatmap_df.set_index(case_id_column)[condition_cols]

    fig = px.imshow(
        heatmap_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=0,
        zmax=1,
        labels=dict(
            x="QCA condition",
            y="Case ID",
            color="Membership"
        ),
        title=title
    )

    fig.update_layout(
        title_x=0.5,
        height=520
    )

    return fig


def create_consistency_coverage_plot(truth_table_df, consistency_cutoff):
    """
    Create Plotly consistency-coverage plot.
    """
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
        height=520,
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05])
    )

    return fig


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Settings")

st.sidebar.subheader("Scoring method")

scoring_method = st.sidebar.radio(
    "Choose text-to-condition scoring method",
    options=["Prototype similarity", "Supervised classifier"],
    index=0,
    help=(
        "Prototype similarity uses Sentence-BERT and conceptual prototypes. "
        "Supervised classifier trains TF-IDF + Logistic Regression models from manually labeled data."
    )
)

# Clear previous results when scoring method changes
if "previous_scoring_method" not in st.session_state:
    st.session_state["previous_scoring_method"] = scoring_method

if st.session_state["previous_scoring_method"] != scoring_method:
    keys_to_clear = [
        "score_df",
        "top_condition_df",
        "threshold_suggestion_df",
        "score_summary_df",
        "fuzzy_df",
        "crisp_df",
        "qca_ready_df",
        "truth_table_df",
        "solution_df",
        "score_fig",
        "fuzzy_heatmap_fig",
        "crisp_heatmap_fig",
        "cc_fig",
        "supervised_validation_report_df",
        "supervised_training_report_df",
        "rich_explanation_df",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["previous_scoring_method"] = scoring_method
    st.info(
        "Scoring method changed. Previous results have been cleared. "
        "Please run the workflow again."
    )

st.sidebar.markdown("---")

if scoring_method == "Prototype similarity":
    model_name = st.sidebar.selectbox(
        "Sentence-BERT model",
        options=[
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v2"
        ],
        index=0,
        help="The default model supports multilingual text, including Chinese and English."
    )
else:
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"

st.sidebar.markdown("---")

st.sidebar.subheader("Calibration settings")

calibration_method = st.sidebar.radio(
    "Calibration method for QCA analysis",
    options=["crisp", "fuzzy"],
    index=0,
    help=(
        "The current basic QCA truth table uses crisp-set conditions. "
        "Fuzzy membership is still shown as an intermediate result."
    )
)

crisp_threshold = st.sidebar.slider(
    "Crisp-set threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.30,
    step=0.01
)

if scoring_method == "Supervised classifier" and crisp_threshold < 0.45:
    st.sidebar.warning(
        "You are using supervised classifier scores, which are probabilities. "
        "A crisp-set threshold around 0.50 is usually more appropriate than 0.30."
    )

full_out = st.sidebar.slider(
    "Fuzzy full-out anchor",
    min_value=0.00,
    max_value=1.00,
    value=0.15,
    step=0.01
)

crossover = st.sidebar.slider(
    "Fuzzy crossover anchor",
    min_value=0.00,
    max_value=1.00,
    value=0.30,
    step=0.01
)

full_in = st.sidebar.slider(
    "Fuzzy full-in anchor",
    min_value=0.00,
    max_value=1.00,
    value=0.45,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.subheader("QCA settings")

consistency_cutoff = st.sidebar.slider(
    "Consistency cutoff",
    min_value=0.00,
    max_value=1.00,
    value=0.80,
    step=0.01
)

min_cases = st.sidebar.number_input(
    "Minimum cases per configuration",
    min_value=1,
    max_value=100,
    value=1,
    step=1
)


# ============================================================
# Main page
# ============================================================

st.title("Text Classification to QCA Analysis Tool") 

st.markdown( 
    """ 
This tool converts raw text into QCA-ready conditions using either prototype-based semantic scoring or supervised classifiers trained on manually labeled data. It supports text-to-condition scoring, fuzzy/crisp calibration, truth-table construction, consistency and coverage calculation, basic solution configuration reporting, reliability diagnostics, visual outputs, and result export. 
"""
)

with st.expander("Expected input format", expanded=False):

    st.markdown(
        """
**Raw text dataset**

This is the target dataset to be scored and converted into QCA-ready cases.  
It is required for both scoring methods.

```csv
case_id,text,outcome
1,我已经多次反映小区垃圾分类设施不足，希望政府尽快处理。,1
2,这个政策解释不清楚，居民不知道应该如何申请。,0
3,我们愿意配合街道办开展社区宣传活动。,1
```
"""
    )

    if scoring_method == "Prototype similarity":
        st.markdown(
            """
**Prototype file**

Prototype similarity mode uses conceptual prototypes to score each raw text.  
Each row defines one QCA condition or outcome concept.

```csv
condition_name,prototype,type
dissatisfaction,The citizen expresses dissatisfaction or complaint. 公民表达不满、抱怨或投诉。,condition
policy_demand,The citizen asks for policy clarification or government action. 公民要求政策解释、政策调整或政府采取行动。,condition
coproduction_request,The citizen shows willingness to cooperate with government. 公民愿意配合政府、参与治理或提供志愿服务。,condition
responsiveness,The government provides clear and concrete response. 政府提供清楚、具体且可执行的回应。,outcome
```
"""
        )

    elif scoring_method == "Supervised classifier":
        st.markdown(
            """
**Manually labeled training data**

Supervised classifier mode uses manually labeled data to train condition classifiers.  
Each QCA condition should be represented by a separate binary 0/1 column.

```csv
case_id,text,dissatisfaction,policy_demand,coproduction_request
L1,居民对停车收费不透明非常不满，认为管理方式不合理。,1,0,0
L2,请问申请补贴需要提交哪些材料？,0,1,0
L3,我们愿意配合社区开展志愿服务。,0,0,1
L4,收费标准不透明，希望政府公开解释依据。,1,1,0
L5,居民愿意参与整改讨论，也希望政策更加清楚。,0,1,1
```

The trained classifiers are then applied to the raw text dataset above.
"""
        )


# ============================================================
# Data upload section
# ============================================================

st.header("1. Upload Data")

if scoring_method == "Prototype similarity":
    col1, col2 = st.columns(2)

    with col1:
        text_file = st.file_uploader(
            "Upload raw text dataset (.csv)",
            type=["csv"],
            key="text_file"
        )

    with col2:
        prototype_file = st.file_uploader(
            "Upload prototype file (.csv)",
            type=["csv"],
            key="prototype_file"
        )

    labeled_file = None

else:
    col1, col2 = st.columns(2)

    with col1:
        text_file = st.file_uploader(
            "Upload raw text dataset (.csv)",
            type=["csv"],
            key="text_file"
        )

    with col2:
        labeled_file = st.file_uploader(
            "Upload manually labeled training data (.csv)",
            type=["csv"],
            key="labeled_file",
            help=(
                "Required when using the supervised classifier. "
                "The file should contain text plus one 0/1 column for each QCA condition."
            )
        )

    prototype_file = None

use_demo = st.checkbox(
    "Use built-in demo files from the data folder",
    value=True
)

text_df = None
prototype_df = None
labeled_df = None

try:
    if use_demo:
        text_df = pd.read_csv("data/demo_text_data.csv")
        prototype_df = pd.read_csv("data/demo_prototypes.csv")

        if os.path.exists("data/supervised_labeled_demo.csv"):
            labeled_df = pd.read_csv("data/supervised_labeled_demo.csv")

        st.success("Demo files loaded from data folder.")
    else:
        if text_file is not None:
            text_df = pd.read_csv(text_file)
        if prototype_file is not None:
            prototype_df = pd.read_csv(prototype_file)
        if labeled_file is not None:
            labeled_df = pd.read_csv(labeled_file)
except Exception as e:
    st.error(f"Failed to load data: {e}")

if text_df is not None:
    st.subheader("Raw text dataset preview")
    st.caption(
        "This is the target dataset to be scored and converted into QCA-ready cases."
    )
    st.dataframe(text_df, use_container_width=True)

if scoring_method == "Prototype similarity":
    if prototype_df is not None:
        st.subheader("Prototype file preview")
        st.caption(
            "Prototype similarity mode uses these conceptual prototypes to score each raw text."
        )
        st.dataframe(prototype_df, use_container_width=True)

        diagnostics = QCADiagnostics()
        prototype_check_df = diagnostics.check_prototype_quality(prototype_df)

        st.subheader("Prototype quality check")
        st.dataframe(prototype_check_df, use_container_width=True)

        if "severity" in prototype_check_df.columns:
            if (prototype_check_df["severity"] == "error").any():
                st.error(
                    "Prototype file contains error-level issues. "
                    "Please fix them before interpreting the results."
                )
            elif (prototype_check_df["severity"] == "warning").any():
                st.warning(
                    "Prototype file contains warning-level issues. "
                    "The workflow can run, but results should be interpreted carefully."
                )
            else:
                st.success("Prototype quality check passed.")
    else:
        st.info("Prototype similarity mode requires a prototype file.")

elif scoring_method == "Supervised classifier":
    if labeled_df is not None:
        st.subheader("Manually labeled training data preview")
        st.caption(
            "Supervised classifier mode uses this labeled dataset to train condition classifiers. "
            "The trained classifiers are then applied to the raw text dataset above."
        )
        st.dataframe(labeled_df, use_container_width=True)
    else:
        st.info("Supervised classifier mode requires manually labeled training data.")

# ============================================================
# Column selection and workflow
# ============================================================

ready_for_workflow = (
    text_df is not None
    and (
        (scoring_method == "Prototype similarity" and prototype_df is not None)
        or
        (scoring_method == "Supervised classifier" and labeled_df is not None)
    )
)

if ready_for_workflow:
    st.header("2. Select Columns")

    available_columns = text_df.columns.tolist()

    default_text_index = available_columns.index("text") if "text" in available_columns else 0
    default_case_index = available_columns.index("case_id") if "case_id" in available_columns else 0
    default_outcome_index = available_columns.index("outcome") if "outcome" in available_columns else 0

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        case_id_column = st.selectbox(
            "Case ID column",
            available_columns,
            index=default_case_index
        )

    with col_b:
        text_column = st.selectbox(
            "Text column",
            available_columns,
            index=default_text_index
        )

    with col_c:
        outcome_column = st.selectbox(
            "Outcome column",
            available_columns,
            index=default_outcome_index
        )

    st.header("3. Run Analysis")

    run_button = st.button("Run full workflow", type="primary")

    if run_button:
        try:
            for key in [
                "supervised_validation_report_df",
                "supervised_training_report_df",
                "rich_explanation_df",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            # --------------------------------------------
            # Step 1: Text scoring
            # --------------------------------------------
            if scoring_method == "Prototype similarity":
                with st.spinner("Encoding texts and prototypes..."):
                    scorer = load_scorer(model_name)

                    score_df = scorer.score_texts(
                        text_df=text_df,
                        prototype_df=prototype_df,
                        text_column=text_column,
                        include_outcome_prototype=False
                    )

                    prototype_explainer = RichTextPrototypeExplainer()

                    rich_explanation_df = prototype_explainer.create_rich_explanation_table(
                        score_df=score_df,
                        prototype_df=prototype_df,
                        text_column=text_column,
                        case_id_column=case_id_column,
                        outcome_column=outcome_column
                    )

                    st.session_state["rich_explanation_df"] = rich_explanation_df

                st.success("Prototype-based text-to-condition scoring and rich explanation completed.")

            else:
                if labeled_df is None:
                    raise ValueError(
                        "Supervised classifier selected, but no manually labeled training data was provided."
                    )

                with st.spinner("Training supervised classifier and predicting condition scores..."):
                    condition_columns = [
                        col for col in labeled_df.columns
                        if col not in ["case_id", text_column, "outcome", "label", "target_label"]
                    ]

                    supervised_clf = SupervisedTextConditionClassifier(
                        text_column=text_column,
                        condition_columns=condition_columns
                    )

                    validation_report_df = supervised_clf.validate_labeled_data(labeled_df)

                    if (validation_report_df["severity"] == "error").any():
                        st.dataframe(validation_report_df, use_container_width=True)
                        raise ValueError(
                            "Labeled training data contains error-level issues. Please fix them before training."
                        )

                    supervised_clf.fit(labeled_df)

                    score_df = supervised_clf.predict_scores(text_df)

                    st.session_state["supervised_validation_report_df"] = validation_report_df
                    st.session_state["supervised_training_report_df"] = supervised_clf.training_report_

                st.success("Supervised text-to-condition scoring completed.")
            # --------------------------------------------
            # Diagnostic step: score explanations and threshold suggestions
            # --------------------------------------------
            diagnostics = QCADiagnostics()

            top_condition_df = diagnostics.create_top_condition_explanation(score_df)

            threshold_suggestion_df, score_summary_df = diagnostics.suggest_thresholds(score_df)

            # --------------------------------------------
            # Step 2: Calibration
            # --------------------------------------------
            calibrator = QCACalibrator()

            fuzzy_df = calibrator.calibrate_dataframe(
                score_df=score_df,
                method="fuzzy",
                full_out=full_out,
                crossover=crossover,
                full_in=full_in,
                keep_score_columns=True
            )

            crisp_df = calibrator.calibrate_dataframe(
                score_df=score_df,
                method="crisp",
                threshold=crisp_threshold,
                keep_score_columns=True
            )

            if calibration_method == "crisp":
                qca_source_df = crisp_df
            else:
                st.warning(
                    "The current truth-table module is designed for crisp-set QCA. "
                    "Fuzzy-set membership is displayed, but the truth table will use crisp-set conditions."
                )
                qca_source_df = crisp_df

            qca_ready_df = calibrator.create_qca_ready_dataset(
                calibrated_df=qca_source_df,
                outcome_column=outcome_column,
                case_id_column=case_id_column
            )

            st.success("Calibration and QCA-ready dataset construction completed.")

            # --------------------------------------------
            # Step 3: QCA truth table
            # --------------------------------------------
            analyzer = BasicQCAAnalyzer(
                consistency_cutoff=consistency_cutoff,
                min_cases=min_cases,
                contradiction_lower=0.40,
                contradiction_upper=0.60
            )

            truth_table_df = analyzer.create_truth_table(
                df=qca_ready_df,
                case_id_column=case_id_column,
                outcome_column=outcome_column
            )

            solution_df = analyzer.extract_solution_configurations(
                truth_table_df
            )

            st.success("Truth table and solution configurations generated.")

            # --------------------------------------------
            # Step 4: Figures
            # --------------------------------------------
            score_fig = create_score_distribution_plot(
                score_df=score_df,
                crisp_threshold=crisp_threshold
            )

            fuzzy_heatmap_fig = create_membership_heatmap(
                membership_df=fuzzy_df,
                title="Figure 2. Fuzzy-Set Membership Heatmap",
                case_id_column=case_id_column,
                outcome_column=outcome_column
            )

            crisp_heatmap_fig = create_membership_heatmap(
                membership_df=crisp_df,
                title="Figure 3. Crisp-Set Membership Heatmap",
                case_id_column=case_id_column,
                outcome_column=outcome_column
            )

            cc_fig = create_consistency_coverage_plot(
                truth_table_df=truth_table_df,
                consistency_cutoff=consistency_cutoff
            )

            # --------------------------------------------
            # Store results in session state
            # --------------------------------------------
            st.session_state["score_df"] = score_df
            st.session_state["top_condition_df"] = top_condition_df
            st.session_state["threshold_suggestion_df"] = threshold_suggestion_df
            st.session_state["score_summary_df"] = score_summary_df

            st.session_state["fuzzy_df"] = fuzzy_df
            st.session_state["crisp_df"] = crisp_df
            st.session_state["qca_ready_df"] = qca_ready_df
            st.session_state["truth_table_df"] = truth_table_df
            st.session_state["solution_df"] = solution_df

            st.session_state["score_fig"] = score_fig
            st.session_state["fuzzy_heatmap_fig"] = fuzzy_heatmap_fig
            st.session_state["crisp_heatmap_fig"] = crisp_heatmap_fig
            st.session_state["cc_fig"] = cc_fig
            # Save the settings used for the current results
            st.session_state["last_run_settings"] = {
                "scoring_method": scoring_method,
                "calibration_method": calibration_method,
                "crisp_threshold": crisp_threshold,
                "full_out": full_out,
                "crossover": crossover,
                "full_in": full_in,
                "consistency_cutoff": consistency_cutoff,
                "min_cases": min_cases,
            }

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# ============================================================
# Results section
# ============================================================

if "score_df" in st.session_state:

    current_settings = {
        "scoring_method": scoring_method,
        "calibration_method": calibration_method,
        "crisp_threshold": crisp_threshold,
        "full_out": full_out,
        "crossover": crossover,
        "full_in": full_in,
        "consistency_cutoff": consistency_cutoff,
        "min_cases": min_cases,
    }

    last_run_settings = st.session_state.get("last_run_settings", None)

    if last_run_settings is not None and current_settings != last_run_settings:
        changed_items = [
            key for key in current_settings
            if current_settings[key] != last_run_settings.get(key)
        ]

        st.warning(
            "The current sidebar settings differ from the settings used to generate "
            "the displayed results. Please click **Run full workflow** again to update "
            "the results. Changed setting(s): "
            + ", ".join(changed_items)
        )

    st.header("4. Intermediate Results and Diagnostics")

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Diagnostics",
            "Condition scores",
            "Fuzzy membership",
            "Crisp membership",
            "QCA-ready dataset"
        ]
    )

    with tab0:
        if "rich_explanation_df" in st.session_state:
            st.subheader("Rich text-prototype match explanations")
            st.markdown(
                """
This table explains why each raw text is matched to its top prototype-based condition.
The matched lexical cues are used only as interpretive aids; the actual condition scores
are still computed from Sentence-BERT embedding similarity.
"""
            )

            st.dataframe(
                st.session_state["rich_explanation_df"],
                use_container_width=True
            )

            st.download_button(
                "Download rich_text_prototype_explanations.csv",
                data=convert_df_to_csv_bytes(st.session_state["rich_explanation_df"]),
                file_name="rich_text_prototype_explanations.csv",
                mime="text/csv"
            )

        if "supervised_validation_report_df" in st.session_state:
            st.subheader("Supervised classifier validation report")
            st.dataframe(
                st.session_state["supervised_validation_report_df"],
                use_container_width=True
            )

        if "supervised_training_report_df" in st.session_state:
            st.subheader("Supervised classifier training report")
            st.dataframe(
                st.session_state["supervised_training_report_df"],
                use_container_width=True
            )

        st.subheader("Top-matched condition explanation")

        st.markdown(
            """
This table shows which conceptual condition each text is most strongly associated with.
The score gap indicates how clearly the top condition is separated from the second-best condition.
A small score gap suggests that the text may be conceptually ambiguous.
"""
        )

        st.dataframe(
            st.session_state["top_condition_df"],
            use_container_width=True
        )

        st.download_button(
            "Download top_condition_explanation.csv",
            data=convert_df_to_csv_bytes(st.session_state["top_condition_df"]),
            file_name="top_condition_explanation.csv",
            mime="text/csv"
        )

        st.subheader("Suggested calibration thresholds")

        st.markdown(
            """
These thresholds are suggested from the empirical distribution of similarity scores.
They are not automatically imposed. Researchers should treat them as references and
make final calibration choices based on theory, data distribution, and substantive knowledge.
"""
        )

        st.dataframe(
            st.session_state["threshold_suggestion_df"],
            use_container_width=True
        )

        st.subheader("Raw score distribution summary")

        st.dataframe(
            st.session_state["score_summary_df"],
            use_container_width=True
        )

    with tab1:
        st.subheader("Condition score table")
        st.caption(
            "In prototype similarity mode, these values are semantic similarity scores. "
            "In supervised classifier mode, these values are predicted condition probabilities."
        )
        st.dataframe(st.session_state["score_df"], use_container_width=True)
        st.download_button(
            "Download score_table.csv",
            data=convert_df_to_csv_bytes(st.session_state["score_df"]),
            file_name="score_table.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Fuzzy calibrated membership table")
        st.dataframe(st.session_state["fuzzy_df"], use_container_width=True)
        st.download_button(
            "Download calibrated_membership_table_fuzzy.csv",
            data=convert_df_to_csv_bytes(st.session_state["fuzzy_df"]),
            file_name="calibrated_membership_table_fuzzy.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Crisp calibrated membership table")
        st.dataframe(st.session_state["crisp_df"], use_container_width=True)
        st.download_button(
            "Download calibrated_membership_table_crisp.csv",
            data=convert_df_to_csv_bytes(st.session_state["crisp_df"]),
            file_name="calibrated_membership_table_crisp.csv",
            mime="text/csv"
        )

    with tab4:
        st.subheader("QCA-ready dataset")
        st.dataframe(st.session_state["qca_ready_df"], use_container_width=True)
        st.download_button(
            "Download qca_ready_dataset.csv",
            data=convert_df_to_csv_bytes(st.session_state["qca_ready_df"]),
            file_name="qca_ready_dataset.csv",
            mime="text/csv"
        )

    st.header("5. QCA Results")

    tab5, tab6 = st.tabs(["Truth table", "Solution configurations"])

    with tab5:
        st.subheader("Truth table")
        st.dataframe(st.session_state["truth_table_df"], use_container_width=True)
        st.download_button(
            "Download truth_table.csv",
            data=convert_df_to_csv_bytes(st.session_state["truth_table_df"]),
            file_name="truth_table.csv",
            mime="text/csv"
        )

    with tab6:
        st.subheader("Solution configurations")
        st.dataframe(st.session_state["solution_df"], use_container_width=True)
        st.download_button(
            "Download solution_configurations.csv",
            data=convert_df_to_csv_bytes(st.session_state["solution_df"]),
            file_name="solution_configurations.csv",
            mime="text/csv"
        )

    st.header("6. Figures")

    fig_tab1, fig_tab2, fig_tab3, fig_tab4 = st.tabs(
        [
            "Score distribution",
            "Fuzzy heatmap",
            "Crisp heatmap",
            "Consistency-coverage"
        ]
    )

    with fig_tab1:
        st.plotly_chart(st.session_state["score_fig"], use_container_width=True)

    with fig_tab2:
        st.plotly_chart(st.session_state["fuzzy_heatmap_fig"], use_container_width=True)

    with fig_tab3:
        st.plotly_chart(st.session_state["crisp_heatmap_fig"], use_container_width=True)

    with fig_tab4:
        st.plotly_chart(st.session_state["cc_fig"], use_container_width=True)

    st.header("7. Download All Results")

    analysis_settings_df = pd.DataFrame([
        {
            "scoring_method": scoring_method,
            "crisp_threshold": crisp_threshold,
            "fuzzy_full_out": full_out,
            "fuzzy_crossover": crossover,
            "fuzzy_full_in": full_in,
            "consistency_cutoff": consistency_cutoff,
            "min_cases": min_cases
        }
    ])

    output_files = {
        "analysis_settings.csv": analysis_settings_df,
        "score_table.csv": st.session_state["score_df"],
        "top_condition_explanation.csv": st.session_state["top_condition_df"],
        "threshold_suggestions.csv": st.session_state["threshold_suggestion_df"],
        "score_distribution_summary.csv": st.session_state["score_summary_df"],
        "calibrated_membership_table_fuzzy.csv": st.session_state["fuzzy_df"],
        "calibrated_membership_table_crisp.csv": st.session_state["crisp_df"],
        "qca_ready_dataset.csv": st.session_state["qca_ready_df"],
        "truth_table.csv": st.session_state["truth_table_df"],
        "solution_configurations.csv": st.session_state["solution_df"],
    }

    if "supervised_validation_report_df" in st.session_state:
        output_files["supervised_validation_report.csv"] = st.session_state[
            "supervised_validation_report_df"
        ]

    if "supervised_training_report_df" in st.session_state:
        output_files["supervised_training_report.csv"] = st.session_state[
            "supervised_training_report_df"
        ]

    if "rich_explanation_df" in st.session_state:
        output_files["rich_text_prototype_explanations.csv"] = st.session_state[
            "rich_explanation_df"
        ]

    output_files["README_result_note.txt"] = (
        "This zip file contains outputs generated by the Text Classification "
        "to QCA Analysis Tool. The workflow includes text-to-condition scoring, "
        "reliability diagnostics, fuzzy/crisp calibration, QCA-ready dataset "
        "construction, truth-table analysis, consistency/coverage calculation, "
        "and solution configuration reporting. If the supervised classifier mode "
        "was used, the zip file also includes supervised classifier validation "
        "and training reports. If the prototype similarity mode was used, the zip "
        "file may also include rich text-prototype match explanations."
    )

    output_zip = create_zip_from_outputs(output_files)

    st.download_button(
        "Download all results as ZIP",
        data=output_zip,
        file_name="text_qca_results.zip",
        mime="application/zip"
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.markdown(
    """
**Method note.** This tool supports two text-to-condition scoring strategies: prototype-based semantic similarity
and supervised classification trained on manually labeled data. Calibration thresholds should be treated as
research-design choices. Researchers should inspect raw scores, membership tables, training diagnostics,
and truth-table results before interpreting configurations substantively.
"""
)
