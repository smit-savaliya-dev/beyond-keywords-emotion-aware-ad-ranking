from src.logic_check import differences, digest, normalised_statements

OLD = '''"""Header."""
from .stats import holm
from src.stats import bootstrap


def f(x):
    """Doc."""
    return holm(x)  # comment


NOTE = "see scripts/40_model_m0_popularity.py and src.stats"
'''


def test_docstrings_comments_and_whitespace_do_not_matter():
    new = OLD.replace('"""Header."""\n', "").replace('"""Doc."""', "").replace("# comment", "# other")
    assert differences(OLD, new) == []
    assert digest(normalised_statements(OLD)) == digest(normalised_statements(OLD + "\n\n\n"))


def test_real_changes_are_reported():
    assert len(differences(OLD, OLD.replace("holm(x)", "holm(x + 1)"))) == 1
    assert differences(OLD, OLD + "\ny = 1\n")


def test_renamed_modules_and_scripts_are_mapped():
    new = (OLD.replace(".stats", ".significance").replace("src.stats", "src.significance")
           .replace("40_model_m0_popularity", "04_01_m0_popularity"))
    assert differences(OLD, new) != []
    assert differences(OLD, new, modules={"stats": "significance"}, stems={"40_model_m0_popularity": "04_01_m0_popularity"}) == []


def test_allowed_statement_may_differ():
    assert differences(OLD, OLD.replace("holm(x)", "holm(x + 1)"), allowed=("f",)) == []
