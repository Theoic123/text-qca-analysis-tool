# Submission Checklist

Before submitting the project, confirm that the following files and functions are present.

## Required files

- [ ] `app.py`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] `technical_note.md`
- [ ] `data/demo_text_data.csv`
- [ ] `data/demo_prototypes.csv`
- [ ] `src/text_scoring.py`
- [ ] `src/calibration.py`
- [ ] `src/qca.py`
- [ ] `src/visualization.py`
- [ ] `src/diagnostics.py`
- [ ] `src/__init__.py`

## Required functions

- [ ] Upload text dataset
- [ ] Upload prototype file
- [ ] Preview raw data
- [ ] Select text, case ID, and outcome columns
- [ ] Score texts against prototypes
- [ ] Show raw score table
- [ ] Perform fuzzy-set calibration
- [ ] Perform crisp-set calibration
- [ ] Produce QCA-ready dataset
- [ ] Generate truth table
- [ ] Report consistency and coverage
- [ ] Report solution configurations
- [ ] Identify contradictory and weak configurations
- [ ] Generate figures
- [ ] Download output files
- [ ] Download all outputs as ZIP

## Recommended test command

```bash
conda activate text_qca
streamlit run app.py
```

Then use the built-in demo files and click `Run full workflow`.
