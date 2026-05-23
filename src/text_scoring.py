import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TextPrototypeScorer:
    """
    Prototype-based text-to-condition scorer.

    This class compares each text case with each conceptual prototype
    and generates semantic similarity scores for QCA condition construction.
    """

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Parameters
        ----------
        model_name : str
            Sentence-BERT model name. The default model supports multilingual text,
            including Chinese and English.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def load_text_data(self, text_file_path):
        """
        Load raw text dataset.

        Expected columns:
        - case_id
        - text
        - outcome

        Parameters
        ----------
        text_file_path : str
            Path to the text CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded text dataset.
        """
        df = pd.read_csv(text_file_path)

        required_columns = ["case_id", "text"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["text"] = df["text"].astype(str)
        return df

    def load_prototypes(self, prototype_file_path):
        """
        Load conceptual prototype dataset.

        Expected columns:
        - condition_name
        - prototype
        - type

        Parameters
        ----------
        prototype_file_path : str
            Path to the prototype CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded prototype dataset.
        """
        prototypes = pd.read_csv(prototype_file_path)

        required_columns = ["condition_name", "prototype", "type"]
        for col in required_columns:
            if col not in prototypes.columns:
                raise ValueError(f"Missing required column: {col}")

        prototypes["condition_name"] = prototypes["condition_name"].astype(str)
        prototypes["prototype"] = prototypes["prototype"].astype(str)
        prototypes["type"] = prototypes["type"].astype(str)

        return prototypes

    def score_texts(
        self,
        text_df,
        prototype_df,
        text_column="text",
        include_outcome_prototype=False
    ):
        """
        Score each text against each conceptual prototype.

        Parameters
        ----------
        text_df : pd.DataFrame
            Raw text data.
        prototype_df : pd.DataFrame
            Prototype data.
        text_column : str
            Name of the text column.
        include_outcome_prototype : bool
            Whether to also score outcome-type prototypes.
            Usually False because outcome is often already given as a variable.

        Returns
        -------
        pd.DataFrame
            Similarity score table.
        """
        if text_column not in text_df.columns:
            raise ValueError(f"Text column '{text_column}' not found in text dataset.")

        if include_outcome_prototype:
            selected_prototypes = prototype_df.copy()
        else:
            selected_prototypes = prototype_df[prototype_df["type"] == "condition"].copy()

        if selected_prototypes.empty:
            raise ValueError("No condition prototypes found. Please check the 'type' column.")

        texts = text_df[text_column].astype(str).tolist()
        prototype_texts = selected_prototypes["prototype"].astype(str).tolist()
        condition_names = selected_prototypes["condition_name"].astype(str).tolist()

        print("Encoding text cases...")
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print("Encoding conceptual prototypes...")
        prototype_embeddings = self.model.encode(
            prototype_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        similarity_matrix = cosine_similarity(text_embeddings, prototype_embeddings)

        score_df = text_df.copy()

        for i, condition_name in enumerate(condition_names):
            score_df[f"{condition_name}_score"] = similarity_matrix[:, i]

        return score_df

    def save_scores(self, score_df, output_path):
        """
        Save similarity score table.

        Parameters
        ----------
        score_df : pd.DataFrame
            Similarity score table.
        output_path : str
            Output CSV file path.
        """
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        score_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Score table saved to: {output_path}")

def run_demo():
    """
    Run demo scoring workflow.
    """
    text_file = "data/demo_text_data.csv"
    prototype_file = "data/demo_prototypes.csv"
    output_file = "outputs/score_table.csv"

    scorer = TextPrototypeScorer()

    text_df = scorer.load_text_data(text_file)
    prototype_df = scorer.load_prototypes(prototype_file)

    score_df = scorer.score_texts(
        text_df=text_df,
        prototype_df=prototype_df,
        text_column="text",
        include_outcome_prototype=False
    )

    print("\nScore table preview:")
    print(score_df.head())

    scorer.save_scores(score_df, output_file)


if __name__ == "__main__":
    run_demo()