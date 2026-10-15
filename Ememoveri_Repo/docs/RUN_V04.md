# v0.4: what is done and what you run with your own credentials

## Done (results in the paper, Section "RQ8")
12 held-out repositories, frozen systems, protocol fixed beforehand (`PROTOCOL_V04.md`).
    python experiments/collect_ext.py ...   (edit repos to HELDOUT_REPOS)  ->  collected.pkl
    python experiments/analyze_heldout.py collected.pkl results/real_ext/summary_ext.json results/real_ext/summary_heldout.json
    python paper/make_heldout.py results/real_ext/summary_heldout.json paper

## Needs a GitHub token: real issue text as the query
    export GITHUB_TOKEN=...                                   # read-only, public repos
    python experiments/issue_tasks.py --git-dir-root repos --repos flask requests httpx rich black poetry pydantic scrapy pytest aiohttp celery sphinx --per-repo 40 --out issue_tasks.json
    python experiments/collect_issue.py --git-dir-root repos --tasks issue_tasks.json --out issue_collected.pkl
    python experiments/analyze_heldout.py issue_collected.pkl results/real_ext/summary_ext.json summary_issues_full.json tfidf:full
    python experiments/analyze_heldout.py issue_collected.pkl results/real_ext/summary_ext.json summary_issues_title.json tfidf:title
Declare hypotheses H1-H3 again before running (copy PROTOCOL_V04.md); expect 20-50% of PRs to link a usable issue.

## Needs an Anthropic API key: retrieval-conditioned patch generation with test execution
    python experiments/export_retrievals.py issue_collected.pkl results/real_ext/summary_ext.json retrievals.json full
    python experiments/agent_eval.py validate --tasks issue_tasks.json --git-dir-root repos --venv-config venvs.json --out validated.json
    ANTHROPIC_API_KEY=... python experiments/agent_eval.py solve --validated validated.json --retrievals retrievals.json --git-dir-root repos --venv-config venvs.json
    python experiments/agent_eval.py report --results agent_results.json --baseline no_memory
`venvs.json`: {"flask": {"python": "envs/flask/bin/python", "pythonpath": ["src", "."]}, ...}; one venv per repository with the test dependencies.
Conditions: no_memory (file tree only), snapshot, sim_state_rec_pop, ememoveri_tuned, oracle (gold files; upper bound).
Validated on real flask commits (tests fail at parent, pass with gold patch). Not yet run with a model. Use >= 100 validated tasks; with 50 tasks only differences above about 0.15 are detectable.
