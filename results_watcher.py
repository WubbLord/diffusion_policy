"""Auto-harvest finished slurm-job eval scores and push to writeup.

For each tracked job:
  - poll squeue until the job leaves the queue
  - read its eval_log.json (or grep test/mean_score from its slurm-N.out)
  - append a dated line to writeup.md's "## Auto-harvested results log" section
  - git commit + push to origin and upstream

Drop-in. Run from the repo root:
    nohup python results_watcher.py > results_watcher.log 2>&1 &

To track more jobs, append to TRACKED below or call --add JOBID:LABEL:eval_log_glob.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, glob
from datetime import datetime
from pathlib import Path

REPO = Path("/data/scratch/sour/DiffusionProject/diffusion_policy")
WRITEUP = REPO / "writeup.md"
STATE = REPO / ".results_watcher_state.json"
POLL_SEC = 120
SECTION = "## Auto-harvested results log"

# (job_id, label_for_log, eval_log_glob_relative_to_repo)
# Use shell-glob; multiple matches OK (each will yield its own log line).
TRACKED = [
    # residual clip=0.05 rerun
    ("828555", "residual clip=0.05 lift kp=1000",
     "data/outputs/2026.05.11/*lift_lowdim_joint_delta_joint5k/eval_latest_residual_kp1000_clip0p05/eval_log.json"),
    ("828556", "residual clip=0.05 can kp=1000",
     "data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k/eval_latest_residual_kp1000_clip0p05/eval_log.json"),
    ("828557", "residual clip=0.05 square kp=3000",
     "data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k/eval_latest_residual_kp3000_clip0p05/eval_log.json"),
    # JP sweep
    ("828541", "JP lift",
     "data/outputs/2026.05.11/*lift_lowdim_joint_delta_joint5k/eval_latest_jp_kp*_dr2.0/eval_log.json"),
    ("828542", "JP can",
     "data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k/eval_latest_jp_kp*_dr2.0/eval_log.json"),
    ("828543", "JP square",
     "data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k/eval_latest_jp_kp*_dr2.0/eval_log.json"),
    # actsteps steps=12 (paths inferred from script convention)
    ("828177", "actsteps sweep",
     "data/outputs/2026.05.11/*_lowdim_joint_delta_joint5k/eval_actsteps_*_kp*/eval_log.json"),
    # demosup square ablation
    ("828539", "demosup_ablate square",
     "data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k/eval_latest_nn_osc_demosup_*/eval_log.json"),
    # BQ Track 2
    ("828211", "BQ NN-OSC can",
     "data/outputs/2026.05.11/*can_lowdim_joint_delta_joint5k/eval_latest_nn_osc_brianquality/eval_log.json"),
    ("828213", "BQ NN-OSC square",
     "data/outputs/2026.05.11/*square_lowdim_joint_delta_joint5k/eval_latest_nn_osc_brianquality/eval_log.json"),
    ("828215", "BQ NN-OSC tool_hang",
     "data/outputs/2026.05.11/*tool_hang_lowdim_joint_delta_joint5k/eval_latest_nn_osc_brianquality/eval_log.json"),
    ("828217", "BQ NN-OSC transport",
     "data/outputs/2026.05.*/*transport_lowdim_joint_delta_joint5k/eval_latest_nn_osc_brianquality/eval_log.json"),
    # transport DP resume (no eval_log, just confirms job completion)
    ("828593", "transport DP resume (5000 ep target)",
     "data/outputs/2026.05.*/*transport_lowdim_joint_delta_joint5k/checkpoints/latest.ckpt"),
]


def sh(*args: str, check: bool = True, cwd: Path = REPO) -> str:
    r = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} → exit {r.returncode}\n{r.stderr}")
    return r.stdout


def squeue_has(job_id: str) -> bool:
    out = sh("squeue", "-h", "-j", job_id, "-o", "%i", check=False)
    return job_id in out


def parse_eval_log(p: Path) -> dict:
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for k in ("test/mean_score", "train/mean_score"):
        if k in d:
            out[k] = d[k]
    return out


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": {}, "harvested_paths": []}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def append_to_writeup(lines: list[str]) -> None:
    text = WRITEUP.read_text()
    if SECTION not in text:
        text = text.rstrip() + "\n\n" + SECTION + "\n"
        text += "Newest entries last. One line per `eval_log.json` (or job completion event).\n\n"
    new_block = "\n".join(lines) + "\n"
    WRITEUP.write_text(text.rstrip() + "\n" + new_block)


def git_push(commit_message: str) -> None:
    sh("git", "add", "writeup.md")
    sh("git", "commit", "-m", commit_message)
    sh("git", "push", "origin", "sour/obs-noise-param")
    sh("git", "push", "upstream", "sour/obs-noise-param")


def harvest_job(job_id: str, label: str, glob_pat: str, state: dict) -> list[str]:
    """Return new log lines (already deduped by path) for this job."""
    paths = sorted(glob.glob(str(REPO / glob_pat)))
    new_lines = []
    for p in paths:
        if p in state["harvested_paths"]:
            continue
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        rel = os.path.relpath(p, REPO)
        if p.endswith(".ckpt"):
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
            new_lines.append(f"- `{ts}` job=**{job_id}** {label} — ckpt `{rel}` saved {mtime}")
        elif p.endswith(".json"):
            scores = parse_eval_log(Path(p))
            score_str = ", ".join(f"{k}={v}" for k, v in scores.items()) or "(no scores)"
            # try to pull the eval-dir name as a hint for which sweep cell this is
            cell = Path(p).parent.name
            new_lines.append(f"- `{ts}` job=**{job_id}** {label} (`{cell}`) — {score_str}")
        else:
            new_lines.append(f"- `{ts}` job=**{job_id}** {label} — file: `{rel}`")
        state["harvested_paths"].append(p)
    return new_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one pass and exit")
    ap.add_argument("--dry-run", action="store_true", help="don't push, just log; ALSO does not modify writeup.md")
    ap.add_argument("--seed", action="store_true", help="mark all currently-existing eval_logs as already-harvested without writing anything; use once at first deployment so the watcher only picks up FUTURE files")
    args = ap.parse_args()

    if args.seed:
        state = load_state()
        for job_id, label, glob_pat in TRACKED:
            paths = sorted(glob.glob(str(REPO / glob_pat)))
            for p in paths:
                if p not in state["harvested_paths"]:
                    state["harvested_paths"].append(p)
        save_state(state)
        print(f"[watcher] seeded {len(state['harvested_paths'])} existing paths into state; future runs will only see new files", flush=True)
        return

    print(f"[watcher] repo={REPO} poll={POLL_SEC}s tracked={len(TRACKED)} jobs", flush=True)
    state = load_state()
    while True:
        accumulated_lines = []
        for job_id, label, glob_pat in TRACKED:
            done = state["done"].get(job_id)
            if done:
                # Job previously seen complete; still harvest any newly-written files
                # (sweep scripts often write multiple eval_logs over time).
                lines = harvest_job(job_id, label, glob_pat, state)
                accumulated_lines.extend(lines)
                continue
            if squeue_has(job_id):
                # still queued / running — try a partial harvest anyway
                lines = harvest_job(job_id, label, glob_pat, state)
                accumulated_lines.extend(lines)
                continue
            # Job left queue → mark done + final harvest
            state["done"][job_id] = datetime.now().isoformat(timespec="seconds")
            lines = harvest_job(job_id, label, glob_pat, state)
            if not lines:
                lines = [f"- `{datetime.now():%Y-%m-%d %H:%M}` job=**{job_id}** {label} — finished, no eval_log found at `{glob_pat}`"]
            accumulated_lines.extend(lines)
            print(f"[watcher] job {job_id} ({label}) finished, harvested {len(lines)} line(s)", flush=True)

        if accumulated_lines:
            if args.dry_run:
                print(f"[watcher] DRY: would append + push {len(accumulated_lines)} line(s):", flush=True)
                for l in accumulated_lines:
                    print(f"  {l}", flush=True)
            else:
                append_to_writeup(accumulated_lines)
                save_state(state)
                msg = f"writeup: auto-harvest {len(accumulated_lines)} eval result line(s)"
                try:
                    git_push(msg)
                    print(f"[watcher] pushed: {msg}", flush=True)
                except Exception as e:
                    print(f"[watcher] git push failed: {e}", flush=True)
        else:
            save_state(state)  # persist any newly-marked-done jobs anyway

        # Exit when all tracked jobs are done and no fresh files appeared this pass.
        remaining = [j for j, *_ in TRACKED if j not in state["done"]]
        if not remaining:
            print("[watcher] all tracked jobs done — exiting", flush=True)
            return
        if args.once:
            print(f"[watcher] --once: exiting with {len(remaining)} job(s) still pending: {remaining}", flush=True)
            return
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
