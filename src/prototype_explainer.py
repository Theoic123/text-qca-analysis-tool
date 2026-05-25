
import re
import pandas as pd
import numpy as np


class RichTextPrototypeExplainer:
    """
    Rich explanation module for prototype-based text-to-condition matching.

    This module does not replace the embedding model. Instead, it explains the
    prototype-similarity scoring results by combining:

    1. top-ranked condition
    2. second-ranked condition
    3. score gap
    4. confidence level
    5. matched lexical cues between the raw text and the prototype
    6. short interpretation notes

    It is designed to be lightweight and deployment-friendly.
    No external tokenizer is required.
    """

    def __init__(self, max_matched_terms=12):
        self.max_matched_terms = max_matched_terms

        self.stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "is", "are", "be", "by", "as", "that", "this", "it", "from",
            "公民", "居民", "政府", "相关", "部门", "进行", "已经", "可以",
            "需要", "希望", "提供", "一个", "这个", "那个", "问题", "当前"
        }

    def normalize_text(self, text):
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_chinese_ngrams(self, text, min_n=2, max_n=4):
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
        ngrams = []

        for chunk in chinese_chunks:
            for n in range(min_n, max_n + 1):
                if len(chunk) >= n:
                    for i in range(len(chunk) - n + 1):
                        ngrams.append(chunk[i:i+n])

        return ngrams

    def extract_english_words(self, text):
        return re.findall(r"[a-zA-Z][a-zA-Z_\-]{2,}", text.lower())

    def extract_explanation_terms(self, text):
        text = self.normalize_text(text)

        terms = []
        terms.extend(self.extract_english_words(text))
        terms.extend(self.extract_chinese_ngrams(text))

        cleaned_terms = []
        for term in terms:
            term = term.strip().lower()
            if not term:
                continue
            if term in self.stopwords:
                continue
            if len(term) < 2:
                continue
            cleaned_terms.append(term)

        return cleaned_terms

    def find_matched_terms(self, text, prototype):
        """
        Find lexical cues shared by the raw text and prototype.

        These matched terms should be interpreted as explanatory cues rather than
        as the actual source of the embedding similarity score.
        """
        text_terms = self.extract_explanation_terms(text)
        prototype_terms = self.extract_explanation_terms(prototype)

        matched = list(set(text_terms).intersection(set(prototype_terms)))
        matched = sorted(matched, key=lambda x: (-len(x), x))

        return matched[:self.max_matched_terms]

    def classify_confidence(self, top_score, second_score):
        if pd.isna(top_score):
            return "unknown"

        if pd.isna(second_score):
            gap = np.nan
        else:
            gap = top_score - second_score

        if top_score >= 0.50 and (pd.isna(gap) or gap >= 0.10):
            return "high"
        elif top_score >= 0.35 and (pd.isna(gap) or gap >= 0.05):
            return "medium"
        elif top_score >= 0.25:
            return "low"
        else:
            return "very low"

    def build_interpretation_note(
        self,
        top_condition,
        top_score,
        second_condition,
        second_score,
        confidence,
        matched_terms
    ):
        if pd.isna(second_score):
            gap_text = "No second condition is available."
        else:
            gap = top_score - second_score
            gap_text = (
                f"The top score exceeds the second-best condition "
                f"({second_condition}) by {gap:.4f}."
            )

        if matched_terms:
            matched_text = "Matched lexical cues include: " + ", ".join(matched_terms) + "."
        else:
            matched_text = (
                "No direct lexical overlap was detected; the match is mainly based on "
                "semantic embedding similarity."
            )

        note = (
            f"The text is most strongly associated with '{top_condition}' "
            f"(score = {top_score:.4f}, confidence = {confidence}). "
            f"{gap_text} {matched_text}"
        )

        return note

    def create_rich_explanation_table(
        self,
        score_df,
        prototype_df,
        text_column="text",
        case_id_column="case_id",
        outcome_column="outcome"
    ):
        score_cols = [col for col in score_df.columns if col.endswith("_score")]

        if not score_cols:
            raise ValueError("No score columns found. Expected columns ending with '_score'.")

        required_proto_cols = ["condition_name", "prototype", "type"]
        for col in required_proto_cols:
            if col not in prototype_df.columns:
                raise ValueError(f"Prototype dataframe missing required column: {col}")

        condition_prototypes = prototype_df[prototype_df["type"] == "condition"].copy()
        prototype_map = dict(
            zip(
                condition_prototypes["condition_name"].astype(str),
                condition_prototypes["prototype"].astype(str)
            )
        )

        rows = []

        for _, row in score_df.iterrows():
            scores = row[score_cols].astype(float).sort_values(ascending=False)

            top_score_col = scores.index[0]
            top_condition = top_score_col.replace("_score", "")
            top_score = float(scores.iloc[0])

            if len(scores) > 1:
                second_score_col = scores.index[1]
                second_condition = second_score_col.replace("_score", "")
                second_score = float(scores.iloc[1])
            else:
                second_condition = None
                second_score = np.nan

            score_gap = top_score - second_score if not pd.isna(second_score) else np.nan
            confidence = self.classify_confidence(top_score, second_score)

            text_value = row[text_column] if text_column in score_df.columns else ""
            top_prototype = prototype_map.get(top_condition, "")

            matched_terms = self.find_matched_terms(text_value, top_prototype)

            interpretation_note = self.build_interpretation_note(
                top_condition=top_condition,
                top_score=top_score,
                second_condition=second_condition,
                second_score=second_score,
                confidence=confidence,
                matched_terms=matched_terms
            )

            result = {
                "case_id": row[case_id_column] if case_id_column in score_df.columns else None,
                "text": text_value,
                "top_condition": top_condition,
                "top_score": round(top_score, 4),
                "second_condition": second_condition,
                "second_score": round(second_score, 4) if not pd.isna(second_score) else np.nan,
                "score_gap": round(score_gap, 4) if not pd.isna(score_gap) else np.nan,
                "confidence": confidence,
                "matched_terms": ", ".join(matched_terms),
                "top_prototype": top_prototype,
                "interpretation_note": interpretation_note
            }

            if outcome_column in score_df.columns:
                result["outcome"] = row[outcome_column]

            rows.append(result)

        return pd.DataFrame(rows)
