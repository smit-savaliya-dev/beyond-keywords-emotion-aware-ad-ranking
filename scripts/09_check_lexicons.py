"""Check the NRC lexicon files: detected format and scale, and VAD lookups for the 27 GoEmotions labels. Writes nothing.

  python scripts/09_check_lexicons.py
"""
import _bootstrap  # noqa: F401
from src.config import GO_VAD_OVERRIDES
from src.emotion import GOEMOTIONS_LABELS
from src.lexicon import load_emolex, load_vad, resolve_terms


def main():
    emolex, _ = load_emolex()
    vad, _ = load_vad()
    unigrams = vad.index[~vad.index.str.contains(" ")]
    labels = [label for label in GOEMOTIONS_LABELS if label != "neutral"]
    mapping, used = resolve_terms(labels, unigrams, GO_VAD_OVERRIDES)
    print(f"\nGoEmotions labels in VAD: {len(labels) - len(used)}/{len(labels)} direct, overrides used: "
          + (", ".join(f"{w} -> {mapping[w]}" for w in used) or "none"))
    print(vad.loc[[mapping[w] for w in labels]].round(3).set_axis(labels).to_string())
    print(f"\nEmoLex words also in VAD unigrams: {emolex.index.isin(unigrams).mean():.1%}")


if __name__ == "__main__":
    main()
