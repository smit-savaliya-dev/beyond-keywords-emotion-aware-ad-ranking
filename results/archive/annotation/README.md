# Archived hand-labelling material

Hand labelling was cancelled (decision D45). Everything here was built for it and was never filled in: 0 of 180 items were labelled.
External validation on SemEval-2007 Task 14 replaced it. Limitation: there are no direct human labels on MIND headlines.

- `manual_sample.csv`: the 180-item sheet. Its ids still exclude those items from `results/tables/emotion_examples_v2.csv`, so that
  table stays unchanged.
- `label_tool.py`, `labelling.py`, `labelling_guidelines.md`, `make_annotation_sheet.py`, `annotation_sheet_functions.py`, `test_labelling.py`:
  the tool, guidelines, sheet maker and tests. They are kept for the record and are not part of the active code or test suite.
