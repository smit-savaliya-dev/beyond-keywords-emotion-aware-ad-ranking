from src.logic_check import differences, digest, normalised_statements

OLD = '''"""Header."""
from .stats import holm
from src.stats import bootstrap


def f(x):
    """Doc."""
    return holm(x)  # comment


NOTE = "see scripts/10_fake_old_name.py and src.stats"
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
           .replace("10_fake_old_name", "01_fake_new_name"))
    assert differences(OLD, new) != []
    assert differences(OLD, new, modules={"stats": "significance"}, stems={"10_fake_old_name": "01_fake_new_name"}) == []


def test_listed_new_statement_is_accepted():
    extra = OLD + "\n\ndef g():\n    return 1\n"
    assert differences(OLD, extra) != []
    assert differences(OLD, extra, added=("g",)) == []


def test_allowed_statement_may_differ():
    assert differences(OLD, OLD.replace("holm(x)", "holm(x + 1)"), allowed=("f",)) == []
