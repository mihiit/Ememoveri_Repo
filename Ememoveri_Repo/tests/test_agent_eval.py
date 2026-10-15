import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import agent_eval as ae

REPLY = """Here is the fix.
<<<<<<< SEARCH pkg/a.py
def f():
    return 1
=======
def f():
    return 2
>>>>>>> REPLACE
<<<<<<< SEARCH pkg/missing.py
x
=======
y
>>>>>>> REPLACE
"""


def test_parse_and_apply():
    b = ae.parse_blocks(REPLY)
    assert [x[0] for x in b] == ["pkg/a.py", "pkg/missing.py"]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "pkg")); open(os.path.join(d, "pkg", "a.py"), "w").write("def f():\n    return 1\n\nprint(f())\n")
        n, errs = ae.apply_blocks(d, b)
        assert n == 1 and errs == ["missing file pkg/missing.py"]
        assert "return 2" in open(os.path.join(d, "pkg", "a.py")).read()


def test_search_text_must_match():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.py"), "w").write("x = 1\n")
        n, errs = ae.apply_blocks(d, [("a.py", "x = 2", "x = 3")])
        assert n == 0 and errs
