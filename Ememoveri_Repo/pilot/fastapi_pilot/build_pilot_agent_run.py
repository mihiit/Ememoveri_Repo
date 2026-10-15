"""
build_pilot_agent_run.py

Step 5 of the E5 pilot: run ONE task through ONE model under ONE memory
condition, combining:
  - the historical source code (checked out at the task's before_commit --
    never the after_commit, which would leak the answer)
  - the mined, leakage-verified memory claims from pilot_memory/<task_id>.json
    (only for the EMEMOVERI condition; NO_MEMORY gets none)
  - a lightweight task specification built from the task's category + note

...and asks the model for a unified diff implementing the change.

This is deliberately a SINGLE call by default -- validate the wiring here
before scaling to the full grid (all tasks x all models x all conditions).

Requires: pip install groq
Requires: GROQ_API_KEY set in your environment
    (PowerShell:  $env:GROQ_API_KEY = "your_key_here"
     bash/mingw:  export GROQ_API_KEY=your_key_here)

Usage:
    # See the exact prompt without calling the API or spending a request:
    python build_pilot_agent_run.py --task fastapi_002 --model openai/gpt-oss-120b --condition EMEMOVERI --dry-run

    # Make the real call (uses the stricter v2 prompt by default -- see
    # PROMPT_VERSIONS below):
    python build_pilot_agent_run.py --task fastapi_002 --model openai/gpt-oss-120b --condition EMEMOVERI

    # Reproduce an earlier v1-prompt run exactly:
    python build_pilot_agent_run.py --task fastapi_002 --model openai/gpt-oss-120b --condition EMEMOVERI --prompt-version v1

Output: pilot_agent_runs/<task_id>__<model_slug>__<condition>__<prompt_version>.json
"""

import argparse
import difflib
import io
import json
import os
import subprocess
import sys
from datetime import datetime

# Windows' console/pipe default encoding (cp1252) cannot represent emoji or
# other non-Latin-1 characters that show up constantly in real commit
# messages and model output (this project has already hit this failure mode
# twice in file-writing code -- this line prevents the same crash from any
# print() call in this script, which was still a latent risk here).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TASKS_PATH = "pilot_tasks.json"
MEMORY_DIR = "pilot_memory"
OUTPUT_DIR = "pilot_agent_runs"
REPO_PATH = "."

MAX_SOURCE_CHARS = 12000    # per-task source budget (sum across all changed files)
MAX_MEMORY_CHARS = 4000     # per-task memory-context budget

# Two documented prompt variants -- NEVER silently edit SYSTEM_PROMPT and
# re-run; add a new version key instead, so every saved result stays
# attributable to the exact instructions that produced it (this is what lets
# you honestly report "a stricter format instruction reduced malformed-patch
# rate from X% to Y%" as a real, comparable finding rather than an
# unreproducible one-off fix).
PROMPT_VERSIONS = {
    "v1": """You are a careful software engineer working on the {repo} codebase.
You will be given:
1. The CURRENT state of one or more source files (as of a specific historical point in time).
2. (Optionally) Relevant engineering memory: prior facts, decisions, or evidence about this codebase.
3. A task description of an engineering change to make.

Produce your answer as a single unified diff (git-style, with `--- a/...` / `+++ b/...` / `@@ ... @@` hunks)
that implements the requested change. Output ONLY the diff -- no prose, no explanation, no markdown fences.
If you cannot determine a correct change from the given information, output exactly: NO_PATCH_POSSIBLE
""",
    "v2": """You are a careful software engineer working on the {repo} codebase.
You will be given:
1. The CURRENT state of one or more source files (as of a specific historical point in time).
2. (Optionally) Relevant engineering memory: prior facts, decisions, or evidence about this codebase.
3. A task description of an engineering change to make.

Produce your answer as a single unified diff in strict git format:
- File headers: `--- a/<path>` and `+++ b/<path>` (use `+++ /dev/null` only for a full file deletion).
- Every hunk header MUST include the full line-number range in the form
  `@@ -<start>,<count> +<start>,<count> @@` -- NEVER output a bare `@@` with no numbers.
  Count every context, added, and removed line in that hunk to compute <count> correctly.
- Every changed/context line must start with a single leading space, `+`, or `-` -- do not add extra
  indentation or omit the leading character.
- Output ONLY the diff -- no prose, no explanation, no markdown code fences before or after it.
If you cannot determine a correct change from the given information, output exactly: NO_PATCH_POSSIBLE
""",
    "v3": """You are a careful software engineer working on the {repo} codebase.
You will be given:
1. The CURRENT state of one or more source files (as of a specific historical point in time).
2. (Optionally) Relevant engineering memory: prior facts, decisions, or evidence about this codebase.
3. A task description of an engineering change to make.

Produce your answer as a single unified diff in strict git format:
- File headers: `--- a/<path>` and `+++ b/<path>` (use `+++ /dev/null` only for a full file deletion).
- Every hunk header MUST include the full line-number range in the form
  `@@ -<start>,<count> +<start>,<count> @@` -- NEVER output a bare `@@` with no numbers.
- Before writing each hunk header, COUNT the lines you are about to output for that hunk:
  <old_count> = number of context lines (lines starting with a single space) PLUS number of
  removed lines (lines starting with `-`) in that hunk, and NOTHING else.
  <new_count> = number of context lines PLUS number of added lines (lines starting with `+`) in
  that hunk, and NOTHING else.
  Double-check these two numbers against the lines you actually wrote before finalizing your answer --
  a header claiming more or fewer lines than the hunk body actually contains will be REJECTED.
- Every changed/context line must start with a single leading space, `+`, or `-` -- do not add extra
  indentation or omit the leading character.
- Output ONLY the diff itself. Do not add any prose, explanation, markdown code fences, or sentinel
  markers (such as "*** End of Patch", "END", or similar) before or after the diff -- the file must
  end immediately after the last diff line.
If you cannot determine a correct change from the given information, output exactly: NO_PATCH_POSSIBLE
""",
}

# A separate contract entirely for --output-mode full_file: the model never
# writes diff syntax at all, eliminating the hunk-arithmetic and sentinel-text
# failure classes by construction. We generate the actual unified diff
# ourselves with Python's difflib against the real historical content, so the
# resulting patch is always syntactically valid.
FULL_FILE_PROMPT = """You are a careful software engineer working on the {repo} codebase.
You will be given:
1. The CURRENT COMPLETE content of one or more source files (as of a specific historical point in time).
2. (Optionally) Relevant engineering memory: prior facts, decisions, or evidence about this codebase.
3. A task description of an engineering change to make.

For EACH file that needs to change, output its COMPLETE new content (the whole file, not just the
changed lines) using exactly this format, once per file:

### FILE: <path exactly as given above>
<the complete new file content, nothing omitted, nothing abbreviated>
### END FILE

If a file should be deleted entirely, output instead:
### DELETE FILE: <path exactly as given above>

Do not include any other text, explanation, or markdown formatting. If multiple files need to change,
repeat the ### FILE / ### END FILE block for each one. If you cannot determine a correct change from
the given information, output exactly: NO_PATCH_POSSIBLE
"""



def run_git(args):
    result = subprocess.run(["git"] + args, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def get_file_at_commit(sha: str, file_path: str) -> str:
    """Read a file's content as it existed at a specific commit -- this is
    how historical source context is obtained WITHOUT ever touching the
    after_commit's version of the file."""
    content = run_git(["show", f"{sha}:{file_path}"])
    if content is None:
        return f"[FILE NOT FOUND AT THIS COMMIT: {file_path} -- possibly newly created after this point]"
    return content


def build_source_context(before_commit: str, changed_files: list, budget: int) -> str:
    sections = []
    remaining = budget
    for file_path in changed_files:
        if remaining <= 0:
            sections.append(f"\n[... remaining files omitted, source budget of {budget} chars exhausted ...]")
            break
        content = get_file_at_commit(before_commit, file_path)
        header = f"\n--- FILE: {file_path} (as of {before_commit[:10]}) ---\n"
        chunk = header + content
        if len(chunk) > remaining:
            chunk = chunk[:remaining] + "\n[... truncated to fit source budget ...]"
        sections.append(chunk)
        remaining -= len(chunk)
    return "".join(sections)


# Known per-model free-tier OUTPUT-tokens-per-minute ceilings discovered
# empirically during this pilot (see the run_pilot_grid.py error logs).
# Requesting max_tokens above one of these caps gets the ENTIRE call
# rejected before generation even starts (a 429 on the *requested* amount,
# not actual usage) -- so capping our own request preemptively converts a
# guaranteed failure into a real attempt.
MODEL_OTPM_CAPS = {
    "qwen/qwen3.8-27b": 950,   # observed OTPM limit: 1000; small safety margin
}


def estimate_max_tokens(before_commit: str, changed_files: list, model: str,
                         floor: int = 512, ceiling: int = 4096) -> int:
    """In full_file mode the model must reproduce roughly as much content as
    it was given (the whole file back, plus a small delta), so total source
    size is a good proxy for required output size -- unlike a fixed
    max_tokens=4096 for every task regardless of whether the file is 50 lines
    or 500. Sizing this adaptively means small tasks stop requesting far more
    headroom than they need, which is exactly what was triggering avoidable
    OTPM rejections for models with small free-tier output budgets."""
    total_chars = sum(len(get_file_at_commit(before_commit, f)) for f in changed_files)
    # ~3.5 chars/token is a reasonable rough estimate for source code; +25%
    # buffer for the ### FILE / ### END FILE markers and minor expansion.
    estimated = int((total_chars / 3.5) * 1.25)
    estimated = max(floor, min(ceiling, estimated))
    model_cap = MODEL_OTPM_CAPS.get(model)
    if model_cap is not None:
        estimated = min(estimated, model_cap)
    return estimated


def build_memory_context(task_id: str, condition: str, budget: int) -> str:
    if condition == "NO_MEMORY":
        return ""
    memory_path = os.path.join(MEMORY_DIR, f"{task_id}.json")
    if not os.path.exists(memory_path):
        print(f"WARNING: no mined memory found at {memory_path} -- run build_pilot_memory.py first. "
              f"Proceeding with empty memory context.", file=sys.stderr)
        return ""
    with open(memory_path, encoding="utf-8") as f:
        memory_data = json.load(f)

    lines = []
    remaining = budget
    for claim in memory_data["claims"]:
        line = (f"- [{claim['state']}, trust={claim['trust']:.2f}] {claim['statement']} "
                f"(scope: {claim['scope']}, {claim['evidence_count']} evidence items)\n")
        if len(line) > remaining:
            break
        lines.append(line)
        remaining -= len(line)
    if not lines:
        return ""
    return "Relevant engineering memory for this codebase area:\n" + "".join(lines)


def build_task_spec(task: dict) -> str:
    return (
        f"Task category: {task['category']}\n"
        f"Task description: {task['note']}\n"
        f"Files involved: {', '.join(task['changed_files'])}\n\n"
        f"Implement this change as a unified diff against the CURRENT file state shown above."
    )


import re

FILE_MARKER_RE = re.compile(r"###\s*FILE:\s*(?P<path>[^\n]+)\n")
DELETE_MARKER_RE = re.compile(r"###\s*DELETE FILE:\s*(?P<path>[^\n]+)")
END_MARKER_RE = re.compile(r"\n?###\s*END FILE\b")


def parse_full_file_response(text: str) -> dict:
    """Parse the full-file-mode model response into {file_path: new_content_or_None}.
    None means the file should be deleted.

    Uses regex search (not line-by-line scanning) specifically to stay robust
    against a real edge case found during testing: if the source file being
    rewritten does not itself end with a trailing newline (common -- e.g.
    Flask's helpers.py), a naive line-by-line parser can fail to recognize
    "### END FILE" as a marker when it lands glued onto the end of the last
    content line with no separating newline.
    """
    files = {}
    pos = 0
    while pos < len(text):
        file_match = FILE_MARKER_RE.search(text, pos)
        delete_match = DELETE_MARKER_RE.search(text, pos)
        if file_match and (not delete_match or file_match.start() <= delete_match.start()):
            path = file_match.group("path").strip()
            content_start = file_match.end()
            end_match = END_MARKER_RE.search(text, content_start)
            if end_match:
                content = text[content_start:end_match.start()]
                pos = end_match.end()
            else:
                content = text[content_start:]
                pos = len(text)
            files[path] = content
        elif delete_match:
            path = delete_match.group("path").strip()
            files[path] = None
            pos = delete_match.end()
        else:
            break
    return files


def _normalize_trailing_newline(content: str) -> str:
    """Collapse any trailing newline variation to exactly one (or none, for
    an empty file). This sidesteps a genuinely ambiguous question -- was a
    trailing newline in the model's response part of the file's real content,
    or just visual separation before the '### END FILE' marker? -- by making
    that distinction not matter: both old and new content are normalized the
    same way before diffing, so the generated diff only reflects real content
    changes, never a spurious trailing-newline artifact."""
    if content == "":
        return content
    return content.rstrip("\n") + "\n"


def generate_mechanical_diff(before_commit: str, parsed_files: dict) -> str:
    """Build a real unified diff with Python's difflib, comparing the actual
    historical content (fetched the same way the prompt's source context
    was built) against the model's proposed new content. This is always
    syntactically valid by construction -- difflib computes correct hunk
    line numbers itself, eliminating the arithmetic-miscount failure class
    entirely."""
    diff_parts = []
    for file_path, new_content in parsed_files.items():
        old_content = _normalize_trailing_newline(get_file_at_commit(before_commit, file_path))
        old_lines = old_content.splitlines(keepends=True)

        if new_content is None:  # deletion
            new_lines = []
            tofile = "/dev/null"
        else:
            new_content = _normalize_trailing_newline(new_content)
            new_lines = new_content.splitlines(keepends=True)
            tofile = f"b/{file_path}"

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{file_path}", tofile=tofile,
            lineterm="\n",
        )
        diff_parts.append("".join(diff))
    return "".join(diff_parts)


def build_prompt(task: dict, condition: str, prompt_version: str, output_mode: str) -> dict:
    source_context = build_source_context(task["before_commit"], task["changed_files"], MAX_SOURCE_CHARS)
    memory_context = build_memory_context(task["task_id"], condition, MAX_MEMORY_CHARS)
    task_spec = build_task_spec(task)

    user_content_parts = [f"## Current source\n{source_context}\n"]
    if memory_context:
        user_content_parts.append(f"\n## Memory context\n{memory_context}\n")
    user_content_parts.append(f"\n## Task\n{task_spec}\n")

    system = PROMPT_VERSIONS[prompt_version].format(repo=task["repo"])
    if output_mode == "full_file":
        system = FULL_FILE_PROMPT.format(repo=task["repo"])
    user = "".join(user_content_parts)
    return {"system": system, "user": user}


def call_groq(model: str, system: str, user: str, temperature: float, max_tokens: int):
    """Returns (raw_output_text, finish_reason). finish_reason should be
    'stop' for a complete response; anything else ('length', etc.) means the
    output may be truncated and any downstream patch-validity failure is
    ambiguous until this is ruled out."""
    try:
        from groq import Groq
    except ImportError:
        print("The groq package is required for real calls. Install it with:\n    pip install groq",
              file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set in your environment. Set it before making a real call:\n"
              "    export GROQ_API_KEY=your_key_here   (bash/mingw)\n"
              "    $env:GROQ_API_KEY = \"your_key_here\"   (PowerShell)", file=sys.stderr)
        sys.exit(1)

    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    finish_reason = response.choices[0].finish_reason
    if finish_reason != "stop":
        print(f"  WARNING: response finish_reason='{finish_reason}' (not 'stop') -- the output may be "
              f"TRUNCATED (hit max_tokens) or cut off for another reason. Do not treat a patch-validity "
              f"failure on this response as a pure model-reasoning error until you've ruled this out.",
              file=sys.stderr)
    return response.choices[0].message.content, finish_reason


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# CPU-only inference needs an explicit context window -- Ollama's default
# (2048 tokens) is smaller than most of this project's prompts, which would
# silently truncate the source context without any error. This is set high
# enough for the source/memory budgets used elsewhere in this script
# (MAX_SOURCE_CHARS + MAX_MEMORY_CHARS, roughly 4000-4500 tokens) plus room
# for the response itself.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))


def call_ollama(model: str, system: str, user: str, temperature: float, max_tokens: int):
    """Returns (raw_output_text, finish_reason), matching call_groq's
    contract so the rest of the pipeline (build_prompt, parsing, validation,
    the grid runner) doesn't need to know which provider served a request.

    Requires a local Ollama server running (`ollama serve`, or the desktop
    app, which runs it automatically) and the model already pulled
    (`ollama pull <model>`). No API key, no rate limits, no deprecation risk
    -- the tradeoff is CPU-only inference is slow per call.
    """
    try:
        import requests
    except ImportError:
        print("The requests package is required for Ollama calls. Install it with:\n    pip install requests",
              file=sys.stderr)
        sys.exit(1)

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=600)
    except requests.exceptions.ConnectionError:
        print(f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is it running? Start it with `ollama serve` "
              f"(or open the Ollama desktop app), and make sure the model is pulled: `ollama pull {model}`",
              file=sys.stderr)
        sys.exit(1)

    if resp.status_code == 404:
        print(f"Ollama returned 404 -- model '{model}' is likely not pulled yet. Run:\n"
              f"    ollama pull {model}", file=sys.stderr)
        sys.exit(1)
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    # Ollama's /api/chat reports done_reason similarly to OpenAI's finish_reason;
    # "length" means max_tokens/num_predict was hit before the model finished.
    finish_reason = data.get("done_reason", "stop" if data.get("done") else "unknown")
    if finish_reason != "stop":
        print(f"  WARNING: done_reason='{finish_reason}' (not 'stop') -- the output may be "
              f"TRUNCATED. Do not treat a patch-validity failure on this response as a pure "
              f"model-reasoning error until you've ruled this out.", file=sys.stderr)
    return content, finish_reason


def call_model(provider: str, model: str, system: str, user: str, temperature: float, max_tokens: int):
    """Single entry point the rest of the script (and the grid runner) calls,
    dispatching to whichever provider was requested."""
    if provider == "ollama":
        return call_ollama(model, system, user, temperature, max_tokens)
    return call_groq(model, system, user, temperature, max_tokens)


def looks_like_a_diff(text: str) -> bool:
    return ("--- a/" in text or "--- " in text) and ("+++ " in text) and ("@@" in text)


def has_well_formed_hunk_headers(text: str) -> bool:
    """A stricter structural check than looks_like_a_diff: every '@@' line
    must actually contain a line-number range, not just the bare marker.
    This directly targets the exact defect seen in the first real pilot run
    (a hunk header of bare '@@' with no '-N,M +N,M')."""
    import re
    hunk_lines = [line for line in text.splitlines() if line.strip().startswith("@@")]
    if not hunk_lines:
        return False
    pattern = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@")
    return all(pattern.match(line.strip()) for line in hunk_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="task_id from pilot_tasks.json, e.g. fastapi_002")
    parser.add_argument("--model", required=True,
                         help="Model ID: a Groq API ID (e.g. openai/gpt-oss-20b) or an Ollama tag "
                              "(e.g. qwen2.5-coder:1.5b) depending on --provider")
    parser.add_argument("--provider", default="groq", choices=["groq", "ollama"],
                         help="'groq' (default): hosted, needs GROQ_API_KEY, subject to rate limits. "
                              "'ollama': local, needs `ollama serve` running and the model pulled -- "
                              "no rate limits, no API key, but CPU-only inference is slow per call.")
    parser.add_argument("--condition", default="EMEMOVERI",
                         choices=["NO_MEMORY", "PLAIN_RAG", "NAIVE_MEMORY", "EMEMOVERI"])
    parser.add_argument("--prompt-version", default="v3", choices=list(PROMPT_VERSIONS.keys()),
                         help="Only used when --output-mode diff. v1 = original loose instructions; "
                              "v2 = strict hunk-header format; v3 (default) = v2 + explicit line-counting "
                              "instruction + forbids sentinel text")
    parser.add_argument("--output-mode", default="full_file", choices=["diff", "full_file"],
                         help="'full_file' (default, recommended): model writes the complete new file "
                              "content and we mechanically generate the diff with difflib -- eliminates "
                              "hunk-arithmetic and sentinel-text failures by construction. 'diff': the "
                              "original approach where the model hand-writes unified-diff syntax directly "
                              "(kept for comparison -- this pilot found a 4/4 hunk-arithmetic failure rate "
                              "on real tasks even with the strictest prompt, v3).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the prompt; do NOT call the API")
    args = parser.parse_args()

    if not os.path.exists(TASKS_PATH):
        print(f"{TASKS_PATH} not found. Run build_pilot_tasks.py first.", file=sys.stderr)
        sys.exit(1)
    with open(TASKS_PATH, encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in json.load(f)}
    if args.task not in tasks:
        print(f"Task '{args.task}' not found in {TASKS_PATH}. Available: {list(tasks.keys())}", file=sys.stderr)
        sys.exit(1)
    task = tasks[args.task]

    prompt = build_prompt(task, args.condition, args.prompt_version, args.output_mode)
    tag = args.prompt_version if args.output_mode == "diff" else "fullfile"

    print(f"=== Prompt for {args.task} | model={args.model} | condition={args.condition} | "
          f"mode={args.output_mode} | prompt={tag} ===")
    print(f"System prompt length: {len(prompt['system'])} chars")
    print(f"User prompt length:   {len(prompt['user'])} chars")

    if args.dry_run:
        print("\n--- SYSTEM ---")
        print(prompt["system"])
        print("\n--- USER ---")
        print(prompt["user"])
        print("\n(dry run -- no API call made)")
        return

    print("\nCalling Groq ...")
    raw_output, finish_reason = call_model(args.provider, args.model, prompt["system"], prompt["user"], args.temperature, args.max_tokens)

    result_record = {
        "task_id": args.task,
        "provider": args.provider,
        "model": args.model,
        "condition": args.condition,
        "output_mode": args.output_mode,
        "prompt_version": tag,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "run_timestamp": datetime.utcnow().isoformat(),
        "finish_reason": finish_reason,
        "system_prompt": prompt["system"],
        "user_prompt": prompt["user"],
        "raw_output": raw_output,
    }

    if args.output_mode == "full_file":
        parsed_files = parse_full_file_response(raw_output)
        if not parsed_files:
            print("\n--- MODEL OUTPUT (no ### FILE / ### DELETE FILE blocks found -- treating as no patch) ---")
            generated_diff = ""
        else:
            generated_diff = generate_mechanical_diff(task["before_commit"], parsed_files)
            print(f"\n--- Parsed {len(parsed_files)} file(s) from model response; "
                  f"generated diff mechanically with difflib ---")
        is_diff_shaped = looks_like_a_diff(generated_diff)
        hunks_well_formed = has_well_formed_hunk_headers(generated_diff) if is_diff_shaped else False
        result_record["parsed_files"] = list(parsed_files.keys())
        result_record["generated_diff"] = generated_diff
        print(generated_diff[:2000] + ("..." if len(generated_diff) > 2000 else ""))
    else:
        is_diff_shaped = looks_like_a_diff(raw_output)
        hunks_well_formed = has_well_formed_hunk_headers(raw_output) if is_diff_shaped else False
        verdict = "looks like a diff, hunk headers well-formed" if hunks_well_formed else (
            "looks like a diff, but hunk headers malformed" if is_diff_shaped else "DOES NOT look like a diff")
        print(f"\n--- MODEL OUTPUT ({verdict}) ---")
        print(raw_output[:2000] + ("..." if len(raw_output) > 2000 else ""))

    result_record["looks_like_a_diff"] = is_diff_shaped
    result_record["hunks_well_formed"] = hunks_well_formed

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{args.task}__{model_slug}__{args.condition}__{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()