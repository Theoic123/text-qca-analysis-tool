# Version 1.2 Extension: Richer Explanation of Text-Prototype Matches

This extension adds a richer explanation layer for the prototype similarity scoring mode.

## What it does

For each raw text, the module reports:

- top-matched condition
- top score
- second-best condition
- second-best score
- score gap
- confidence level
- matched lexical cues between raw text and prototype
- prototype text used for the top condition
- natural-language interpretation note

## Important note

The matched lexical cues are explanatory aids. They are not the actual mathematical basis of the Sentence-BERT similarity score. The actual score is still computed using embedding-based cosine similarity.

## Output file

The recommended output file name is:

```text
rich_text_prototype_explanations.csv
```

## Recommended UI placement

Display this table in the Diagnostics tab only when the user selects:

```text
Prototype similarity
```

For supervised classifier mode, this explanation is not applicable because no prototype is used for scoring.
