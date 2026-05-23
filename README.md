# Text Classification to QCA Analysis Tool

## 1. Project Overview

This project is a research-oriented web tool that converts raw text into QCA-ready condition variables and produces basic crisp-set QCA outputs. It is designed for researchers working with citizen messages, government replies, policy consultation comments, or other short administrative and governance-related texts.

The workflow is:

1. Upload a raw text dataset.
2. Upload conceptual prototype descriptions.
3. Score each text against each prototype using multilingual Sentence-BERT.
4. Convert semantic similarity scores into fuzzy-set and crisp-set membership values.
5. Generate a QCA-ready dataset.
6. Produce a truth table with consistency and coverage.
7. Identify sufficient, contradictory, and weak configurations.
8. Generate diagnostic tables and interpretation-oriented figures.
9. Export all main outputs as CSV files.

The tool is built with Python and Streamlit. The user does not need to manually edit code to run the main workflow.

---

## 2. Folder Structure

```text
text_qca_tool/
│
├── app.py
├── requirements.txt
├── README.md
├── technical_note.md
│
├── data/
│   ├── demo_text_data.csv
│   └── demo_prototypes.csv
│
├── outputs/
│   ├── score_table.csv
│   ├── calibrated_membership_table_fuzzy.csv
│   ├── calibrated_membership_table_crisp.csv
│   ├── qca_ready_dataset.csv
│   ├── truth_table.csv
│   ├── solution_configurations.csv
│   ├── top_condition_explanation.csv
│   ├── threshold_suggestions.csv
│   └── score_distribution_summary.csv
│
└── src/
    ├── __init__.py
    ├── text_scoring.py
    ├── calibration.py
    ├── qca.py
    ├── visualization.py
    └── diagnostics.py
```

---

## 3. Installation

### Step 1: Create a Conda environment

```bash
conda create -n text_qca python=3.10 -y
conda activate text_qca
```

### Step 2: Install required packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not used, install packages manually:

```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers plotly openpyxl kaleido
```

---

## 4. Run the Web App

From the project root folder, run:

```bash
streamlit run app.py
```

The app will open in the browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 5. Input Data Format

### 5.1 Text Dataset

The text dataset should be a CSV file. A minimal example:

```csv
case_id,text,outcome
1,我已经多次反映小区垃圾分类设施不足，希望政府尽快处理。,1
2,这个政策解释不清楚，居民不知道应该如何申请。,0
3,我们愿意配合街道办开展社区宣传活动。,1
```

Required fields:

| Column | Description |
|---|---|
| `case_id` | Unique case identifier |
| `text` | Raw text to be classified |
| `outcome` | Binary outcome variable, usually 0 or 1 |

The app allows the user to select alternative column names through the interface.

### 5.2 Prototype File

The prototype file should be a CSV file. A minimal example:

```csv
condition_name,prototype,type
dissatisfaction,The citizen complains about a public problem or expresses dissatisfaction. 公民表达不满、抱怨、投诉、失望或认为处理方式不合理。,condition
policy_demand,The citizen asks for policy clarification, adjustment, implementation, or government action. 公民要求政策解释、政策调整、明确申请条件或政府采取具体行动。,condition
coproduction_request,The citizen shows willingness to cooperate, participate, volunteer, or co-produce public services. 公民愿意配合政府、参与社区治理、提供志愿服务或共同完成公共服务。,condition
responsiveness,The government provides a clear, concrete, respectful, and actionable response. 政府提供清楚、具体、尊重且可执行的回应。,outcome
```

Required fields:

| Column | Description |
|---|---|
| `condition_name` | Name of the QCA condition |
| `prototype` | Conceptual description used for semantic scoring |
| `type` | Either `condition` or `outcome` |

In the current workflow, rows with `type = condition` are used for text-to-condition scoring. The outcome variable is taken from the uploaded text dataset.

---

## 6. Main Workflow

### Step 1: Load data

The user may either use the built-in demo files in the `data/` folder or upload their own CSV files.

### Step 2: Check prototypes

The app checks whether the prototype file contains required columns, duplicated condition names, invalid types, empty values, and overly short or long prototype descriptions.

### Step 3: Score texts against prototypes

The app uses a multilingual Sentence-BERT model to compute semantic similarity between each text and each conceptual prototype. The default model is:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

The result is a similarity score table.

### Step 4: Calibrate scores

The tool supports both fuzzy-set and crisp-set calibration.

Default fuzzy-set anchors:

```text
full_out = 0.15
crossover = 0.30
full_in = 0.45
```

Default crisp-set threshold:

```text
threshold = 0.30
```

These values can be changed in the sidebar.

### Step 5: Generate QCA results

The tool generates:

- QCA-ready dataset
- truth table
- consistency values
- raw coverage values
- sufficient configurations
- contradictory configurations
- weak configurations

### Step 6: Inspect figures and diagnostics

The app provides:

- score distribution plot
- fuzzy-set membership heatmap
- crisp-set membership heatmap
- consistency-coverage plot
- top-matched condition explanation
- threshold suggestion table
- raw score distribution summary

### Step 7: Export results

The user can download individual CSV files or download all main results as a ZIP file.

---

## 7. Output Files

| Output file | Description |
|---|---|
| `score_table.csv` | Raw semantic similarity scores |
| `top_condition_explanation.csv` | Highest-scoring condition for each text |
| `threshold_suggestions.csv` | Suggested calibration thresholds from the score distribution |
| `score_distribution_summary.csv` | Summary statistics of raw similarity scores |
| `calibrated_membership_table_fuzzy.csv` | Fuzzy-set membership values |
| `calibrated_membership_table_crisp.csv` | Crisp-set membership values |
| `qca_ready_dataset.csv` | Dataset ready for QCA truth-table analysis |
| `truth_table.csv` | Observed configurations with consistency and coverage |
| `solution_configurations.csv` | Sufficient configurations selected from the truth table |

---

## 8. Interpretation Notes

The tool is intended to support research judgment, not replace it. Prototype design, calibration thresholds, and outcome definition should be grounded in the research question.

A high semantic similarity score means that a text is conceptually close to a prototype. A calibrated membership value indicates whether the text is considered in or out of a QCA condition set. A sufficient configuration means that cases sharing the same condition combination are consistently associated with the outcome.

The current QCA module focuses on crisp-set truth-table analysis. Fuzzy-set membership values are generated and visualized, but full fuzzy-set Boolean minimization is treated as a future extension.

---

## 9. Reproducibility

To reproduce the demo results:

```bash
conda activate text_qca
streamlit run app.py
```

Then select:

```text
Use built-in demo files from the data folder
```

and click:

```text
Run full workflow
```

The app should produce the same intermediate tables, QCA outputs, and figures.

---

## 10. Limitations

1. Prototype-based scoring depends on the quality of conceptual prototypes.
2. Short or ambiguous texts may produce unstable similarity scores.
3. Calibration thresholds are research-design choices and should be inspected carefully.
4. The current truth-table module is designed for crisp-set QCA.
5. The tool does not replace specialized QCA software for advanced minimization.
6. Small samples may produce unstable configurations.

---

## 11. Future Improvements

Possible extensions include:

- supervised text classifiers trained on manually labeled data
- richer explanation of text-prototype matches
- threshold sensitivity analysis
- fuzzy-set QCA minimization
- contradiction-resolving tools
- multilingual prototype templates
- direct export to R QCA package format
