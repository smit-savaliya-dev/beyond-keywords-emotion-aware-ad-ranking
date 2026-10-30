"""Optional in-domain check: the classifier + 28->6 mapping on the GoEmotions test split."""
import json

from sklearn.metrics import accuracy_score, classification_report, f1_score

import _bootstrap  # noqa: F401
from src.config import EMOTION_MAX_LENGTH, EMOTION_MODEL, RESULTS
from src.emotion import SIX_CLASSES, collapse, load_classifier, pick_device, score_texts, single_label_gold


def main():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    idx, gold = single_label_gold(ds["labels"], ds.features["labels"].feature.names)
    texts = [ds["text"][i] for i in idx]
    clf = load_classifier(EMOTION_MODEL, pick_device("auto"))
    pred = [SIX_CLASSES[int(j)] for j in collapse(score_texts(clf, texts, 64, EMOTION_MAX_LENGTH)).argmax(axis=1)]
    res = {"n_single_label_items": len(gold), "accuracy": accuracy_score(gold, pred),
           "macro_f1": f1_score(gold, pred, labels=SIX_CLASSES, average="macro", zero_division=0)}
    print(res)
    print(classification_report(gold, pred, labels=SIX_CLASSES, zero_division=0))
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables" / "emotion_in_domain.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
