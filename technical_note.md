# Technical Note: Text Classification to QCA Analysis Tool

## 1\. What the Tool Does

This tool converts raw text into QCA-ready condition variables and produces basic configurational analysis outputs. It is designed for researchers who work with citizen messages, government replies, policy consultation comments, complaint texts, or other short governance-related documents. The tool operationalizes conceptual categories, such as dissatisfaction, policy demand, co-production request, and government responsiveness, by comparing each text with researcher-defined prototype descriptions.

The core workflow has four stages. First, the user uploads a text dataset and a prototype file. Second, the tool uses a multilingual Sentence-BERT model to calculate semantic similarity between each text and each conceptual prototype. Third, the raw similarity scores are calibrated into fuzzy-set and crisp-set membership values. Fourth, the crisp-set conditions are used to construct a QCA truth table, calculate consistency and raw coverage, and identify sufficient, weak, and contradictory configurations.

The tool is implemented as a Streamlit web application. The main workflow can be completed through the user interface without editing code.

## 2\. Required Data

The tool requires two CSV files.

The first file is a raw text dataset. It should contain one row per case, a text column, a case identifier, and a binary outcome column. The default expected columns are `case\_id`, `text`, and `outcome`, although the interface allows the user to select alternative column names.

The second file is a prototype file. It should contain `condition\_name`, `prototype`, and `type`. Rows marked as `condition` are used to score texts. Rows marked as `outcome` are retained as conceptual information, but the actual outcome variable is taken from the uploaded text dataset.

## 3\. Method

The text-scoring step uses prototype-based semantic similarity. Each raw text and each conceptual prototype is embedded with a multilingual Sentence-BERT model. The tool then calculates cosine similarity between text embeddings and prototype embeddings. These similarity values are interpreted as raw condition scores.

The calibration step supports both fuzzy-set and crisp-set logic. In fuzzy-set calibration, raw scores are transformed into membership values between 0 and 1 using three anchors: full-out, crossover, and full-in. The default values are 0.15, 0.30, and 0.45. In crisp-set calibration, a case is assigned to a condition if its similarity score is greater than or equal to the selected threshold. The default crisp-set threshold is 0.30.

The QCA module currently focuses on crisp-set truth-table analysis. For each observed condition configuration, the tool computes consistency as the proportion of cases in that configuration with outcome equal to 1. Raw coverage is calculated as the number of outcome-positive cases covered by the configuration divided by the total number of outcome-positive cases. Configurations above the user-selected consistency cutoff are reported as sufficient configurations. Configurations with intermediate consistency values are marked as contradictory, while low-consistency configurations are marked as weak.

## 4\. Outputs

The tool produces several outputs. The similarity score table records the raw semantic similarity between each text and each prototype. The fuzzy and crisp membership tables show how raw scores are converted into set-membership values. The QCA-ready dataset contains one row per case, one column per calibrated condition, and one outcome column. The truth table reports observed configurations, number of cases, outcome-positive cases, consistency, raw coverage, case IDs, and configuration status. The solution configuration table reports configurations that meet the consistency and case-count criteria.

The app also provides reliability diagnostics. A prototype quality check flags missing columns, duplicate condition names, invalid types, empty prototypes, and overly short or long prototypes. A top-condition explanation table reports the strongest and second-strongest condition match for each case. A threshold suggestion table summarizes possible calibration thresholds based on the empirical distribution of similarity scores.

The tool also generates visual outputs: a raw score distribution plot, fuzzy-set membership heatmap, crisp-set membership heatmap, and consistency-coverage plot.

## 5\. Interpretation

The results should be interpreted as research-support outputs rather than fully automatic classifications. A high similarity score indicates that a text is semantically close to a conceptual prototype. A calibrated membership value indicates whether the case is considered in or out of a condition set. A sufficient configuration suggests that the observed condition combination is consistently associated with the outcome in the uploaded data.

However, QCA interpretation requires substantive judgment. Researchers should inspect the raw texts, prototype descriptions, score tables, and calibration thresholds before drawing conclusions. Contradictory configurations are especially important because they indicate that the same condition combination is associated with different outcome values across cases.

## 6\. Assumptions

The tool assumes that the conceptual prototypes are meaningful and sufficiently specific. It also assumes that semantic similarity is an acceptable proxy for condition membership. The calibration thresholds are treated as researcher-controlled design choices. The current QCA module assumes crisp-set conditions for truth-table construction. Fuzzy-set membership values are generated for inspection, but full fuzzy-set QCA minimization is not yet implemented.

## 7\. Limitations

The main limitation is that prototype-based scoring depends heavily on prototype quality. Short, vague, or overlapping prototypes may produce unstable scores. Short texts may also be difficult to classify because they provide limited semantic context. Calibration choices can strongly affect QCA results, especially in small samples. The current solution configuration module reports sufficient observed configurations but does not perform advanced Boolean minimization. Therefore, the tool should not be treated as a replacement for specialized QCA software.

Another limitation is that the default model is general-purpose rather than trained specifically on governance texts. For applied research, manually labeled validation data would improve reliability.

## 8\. Future Improvements

With more time, the tool could be extended in several directions. First, it could include threshold sensitivity analysis to show how truth-table results change under different calibration cutoffs. Second, it could support supervised text classification using manually labeled training data. Third, it could implement fuzzy-set QCA minimization and export formats compatible with R QCA packages. Fourth, it could provide richer case-level explanations by highlighting text segments most relevant to each prototype. Finally, it could include contradiction-resolution tools to help researchers inspect and refine ambiguous configurations.

