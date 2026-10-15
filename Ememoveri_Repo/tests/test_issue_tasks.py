import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import issue_tasks as it


def test_regexes():
    assert it.PR_SUBJ.search("Fix header parsing (#1234)").group(1) == "1234"
    assert it.PR_SUBJ.search("Merge branch main") is None
    assert [int(m) for m in it.CLOSES.findall("This fixes #12, closes psf/requests#34 and Resolves: https://github.com/a/b/issues/56")] == [12, 34, 56]
    assert it.CLOSES.findall("see #99 for context") == []


def test_clean_text():
    t = it.clean_text("<!-- template -->Bug: crash\n\n\n\n![img](x.png) on   start", 100)
    assert "template" not in t and "img" not in t and "\n\n\n" not in t and "  " not in t
