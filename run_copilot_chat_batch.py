#!/usr/bin/env python3
"""
run_copilot_chat_batch.py

Batch-ask GitHub Copilot CLI (Copilot Chat in terminal) a list of questions across multiple repos
and store responses in a CSV.

Requirements:
- GitHub Copilot CLI installed (copilot)
- You have authenticated in an interactive session at least once (e.g., run `copilot` and `/login`)
- git installed (if cloning is enabled)

Notes:
- Copilot CLI is agentic and may ask for approvals/trust prompts the first time it sees a directory.
  For smooth runs, open each cloned repo once interactively and "trust" it.
- We invoke Copilot CLI in scripting mode using: copilot -p "<prompt>" (as documented in GitHub changelog).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


# -----------------------------
# Inputs
# -----------------------------
REPO_URLS = [
    "https://github.com/aws/aws-pgsql-odbc",
    "https://github.com/Netflix/hollow",
    "https://github.com/microsoft/lisa",
    "https://github.com/google/meridian",
    "https://github.com/apple/pkl",
]

QUESTIONS = [
    "Who are the top 5 contributors by commit count?",
    "Who are the top 10 contributors over the past 6 months?",
    "Which five files have been modified the most?",
    "How many unique contributors are there?",
    "Who is the most active contributor?",
    "Which author has contributed the most to the documentation (Markdown files)?",
    "Which issue has the highest number of comments?",
    "Who is the most active issue reporter?",
    "Which issue is the oldest open issue?",
    "What is the average issue resolution time?",
    "How do I contribute to this project?",
    "What is the project’s code of conduct?",
    "How do I report a security vulnerability?",
    "What steps should I follow before opening a pull request?",
    "How are major decisions made in this project?",
    "What skills or tools are needed to contribute?",
    "How can I find a good first issue to work on?",
    "Are there contribution guidelines for this project?",
]


# -----------------------------
# Helpers
# -----------------------------
def repo_name_from_url(url: str) -> str:
    # https://github.com/org/name -> name
    m = re.search(r"github\.com/[^/]+/([^/#?]+)", url)
    return m.group(1) if m else url.rstrip("/").split("/")[-1]


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_s: int = 1800,
) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return p.returncode, p.stdout, p.stderr


def ensure_repo_cloned(repo_url: str, workspace: Path, depth: int = 1) -> Path:
    """Clone repo if missing; return local path."""
    name = repo_name_from_url(repo_url)
    dest = workspace / name
    if dest.exists() and (dest / ".git").exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), repo_url, str(dest)]
    rc, out, err = run_cmd(cmd, cwd=workspace, timeout_s=3600)
    if rc != 0:
        raise RuntimeError(f"git clone failed for {repo_url}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return dest


def copilot_prompt_for(repo_name: str, repo_url: str, question: str) -> str:
    """
    Your requested inference template:
      For this repo {repo_name} ({repo_url}), answer this question - "{question}"
    """
    return f'For this repo {repo_name} ({repo_url}), answer this question - "{question}"'


def call_copilot_cli(
    prompt: str,
    cwd: Path,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    timeout_s: int = 1200,
    silent: bool = True,
) -> str:
    """
    Invoke Copilot CLI in scripted mode:
      copilot -p "<prompt>" [--silent] [--model <name>]

    We keep flags minimal so it’s more likely to work across versions.
    """
    cmd = [copilot_bin, "-p", prompt]
    if silent:
        cmd.append("--silent")
    if model:
        cmd += ["--model", model]

    rc, out, err = run_cmd(cmd, cwd=cwd, timeout_s=timeout_s)

    # Some builds may write useful info to stderr; we keep it if stdout is empty.
    text = out.strip()
    if not text and err.strip():
        text = err.strip()

    if rc != 0:
        raise RuntimeError(
            f"copilot CLI failed (rc={rc})\nPROMPT:\n{prompt}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"
        )

    return text


def append_csv_row(csv_path: Path, header: List[str], row: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="copilot_responses.csv", help="Output CSV path")
    ap.add_argument("--workspace", type=str, default="copilot_workspace", help="Where to clone repos")
    ap.add_argument(
        "--no_clone",
        action="store_true",
        help="Do not clone repos; run copilot from current working directory instead (less accurate).",
    )
    ap.add_argument(
        "--copilot_bin",
        type=str,
        default=os.environ.get("COPILOT_BIN", "copilot"),
        help="Copilot CLI binary name/path (default: 'copilot')",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name passed to copilot via --model (if supported by your installation).",
    )
    ap.add_argument(
        "--sleep_s",
        type=float,
        default=0.5,
        help="Sleep between calls to reduce rate/throughput issues",
    )
    ap.add_argument(
        "--timeout_s",
        type=int,
        default=1200,
        help="Timeout per copilot call (seconds)",
    )
    args = ap.parse_args()

    out_csv = Path(args.out_csv).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()

    header = [
        "timestamp_utc",
        "repo_name",
        "repo_url",
        "question_id",
        "question",
        "prompt",
        "response",
        "copilot_bin",
        "model",
        "local_repo_path",
    ]

    for repo_url in REPO_URLS:
        name = repo_name_from_url(repo_url)

        if args.no_clone:
            repo_path = Path.cwd()
        else:
            repo_path = ensure_repo_cloned(repo_url, workspace=workspace, depth=1)

        for qi, q in enumerate(QUESTIONS, start=1):
            ts = datetime.now(timezone.utc).isoformat()
            prompt = copilot_prompt_for(name, repo_url, q)

            try:
                resp = call_copilot_cli(
                    prompt=prompt,
                    cwd=repo_path,
                    copilot_bin=args.copilot_bin,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    silent=True,
                )
                status_resp = resp
            except Exception as e:
                # Record failure in CSV (still useful for analysis)
                status_resp = f"[ERROR] {e}"

            append_csv_row(
                out_csv,
                header=header,
                row=[
                    ts,
                    name,
                    repo_url,
                    str(qi),
                    q,
                    prompt,
                    status_resp,
                    args.copilot_bin,
                    args.model or "",
                    str(repo_path),
                ],
            )

            time.sleep(args.sleep_s)

    print(f"Done. Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
