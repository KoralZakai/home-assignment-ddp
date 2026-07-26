"""Parse the 1-GPU and 4-GPU training logs into the report's graphs and tables.

Usage:
    python make_report_data.py logs/1gpu_log.txt logs/4gpu_log.txt
    python make_report_data.py --selftest      # runs on a synthetic log, no real run needed

Produces graphs/loss_vs_steps.png, graphs/loss_vs_time.png and prints the three
markdown tables (training throughput, inference throughput, communication).
"""
import re
import sys
from pathlib import Path

# HF Trainer prints its logs as a python dict repr, e.g.
#   {'loss': 10.91, 'grad_norm': 1.2, 'learning_rate': 1.9e-06, 'epoch': 0.05}
# `step` is not always in that dict, so we fall back to inferring it from
# logging_steps (the Trainer emits exactly one loss line per logging_steps).
NUM = r"[-+0-9.eE]+"
RE_LOSS = re.compile(r"'loss':\s*'?(" + NUM + r")")
RE_STEP_IN_DICT = re.compile(r"'step':\s*(\d+)")
RE_COMM_STEP = re.compile(
    r"\[Comm\]\s+step=(\d+)\s+avg_step_time_s=(" + NUM + r")"
    r"\s+cumulative_comm_time_s=(" + NUM + r")"
    r"\s+cumulative_comm_bytes=(\d+)"
)
RE_METRIC = re.compile(r"'(train_runtime|train_samples_per_second|train_steps_per_second|"
                       r"eval_samples_per_second|eval_runtime|eval_loss)':\s*'?(" + NUM + r")")
RE_INFERENCE = re.compile(r"\[Inference\]\s+(\w+):\s*(" + NUM + r")")
# NB: train.py prints the ring-bytes line with *no* space after the colon,
# so the separator must be \s* and not \s+.
RE_COMM_SUMMARY = re.compile(r"\[Comm\]\s+(.+?):\s*([-0-9.,]+)")
RE_PARAMS = re.compile(r"\[Comm\] Trainable parameters:\s*([0-9,]+)")


def parse(path, logging_steps=10):
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # --- loss curve -----------------------------------------------------
    losses = []  # (step, loss)
    for i, line in enumerate(text.splitlines()):
        if "'loss'" not in line or "'eval_loss'" in line:
            continue
        m = RE_LOSS.search(line)
        if not m:
            continue
        s = RE_STEP_IN_DICT.search(line)
        step = int(s.group(1)) if s else (len(losses) + 1) * logging_steps
        losses.append((step, float(m.group(1))))

    # --- per-step timing from the [Comm] lines --------------------------
    comm_steps = [(int(a), float(b), float(c), int(d))
                  for a, b, c, d in RE_COMM_STEP.findall(text)]

    # cumulative wall-clock time at each [Comm] step
    times, t, prev_step = {}, 0.0, 0
    for step, avg_step_time, _, _ in comm_steps:
        t += avg_step_time * (step - prev_step)
        times[step] = t
        prev_step = step

    # --- final metrics --------------------------------------------------
    metrics = {}
    for key, val in RE_METRIC.findall(text):
        metrics[key] = float(val)          # last occurrence wins == final eval
    for key, val in RE_INFERENCE.findall(text):
        metrics[key] = float(val)          # [Inference] block overrides

    # --- communication summary -----------------------------------------
    summary = {}
    tail = text.split("[Comm] Communication summary")[-1] if "Communication summary" in text else ""
    for key, val in RE_COMM_SUMMARY.findall(tail):
        summary[key.strip()] = val.strip()

    params = RE_PARAMS.search(text)
    return {
        "path": str(path),
        "losses": losses,
        "comm_steps": comm_steps,
        "times": times,
        "metrics": metrics,
        "summary": summary,
        "params": params.group(1) if params else "n/a",
        "total_time_s": t,
    }


def loss_vs_time(run):
    """Map each logged loss point onto wall-clock seconds.

    The [Comm] lines only land every logging_steps, so interpolate linearly
    between them; fall back to train_runtime scaling if no [Comm] lines exist.
    """
    pts = sorted(run["times"].items())
    if not pts:
        runtime = run["metrics"].get("train_runtime")
        last = run["losses"][-1][0] if run["losses"] else 1
        if not runtime:
            return []
        return [(step / last * runtime, loss) for step, loss in run["losses"]]

    out = []
    for step, loss in run["losses"]:
        prev = (0, 0.0)
        for s, tt in pts:
            if s <= step:
                prev = (s, tt)
            else:
                frac = (step - prev[0]) / (s - prev[0])
                out.append((prev[1] + frac * (tt - prev[1]), loss))
                break
        else:
            # past the last [Comm] line: extrapolate with the last step rate
            rate = pts[-1][1] / pts[-1][0] if pts[-1][0] else 0
            out.append((pts[-1][1] + (step - pts[-1][0]) * rate, loss))
    return out


def plot(runs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run in runs.items():
        xs = [s for s, _ in run["losses"]]
        ys = [l for _, l in run["losses"]]
        ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set(xlabel="training step", ylabel="training loss",
           title="GPT-2 Large / WikiText-2 — loss vs. steps")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_vs_steps.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run in runs.items():
        pts = loss_vs_time(run)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", markersize=3, label=label)
    ax.set(xlabel="wall-clock time (s)", ylabel="training loss",
           title="GPT-2 Large / WikiText-2 — loss vs. wall-clock time (fair comparison)")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_vs_time.png", dpi=150)
    return [outdir / "loss_vs_steps.png", outdir / "loss_vs_time.png"]


def table(title, rows, runs):
    labels = list(runs)
    out = [f"\n**{title}**\n", "| Metric | " + " | ".join(labels) + " |",
           "|---|" + "---|" * len(labels)]
    for label, getter in rows:
        cells = [str(getter(runs[l])) for l in labels]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def report(runs):
    def metric(key, fmt="{:.4f}"):
        return lambda r: fmt.format(r["metrics"][key]) if key in r["metrics"] else "n/a"

    def summ(key):
        return lambda r: r["summary"].get(key, "n/a")

    print(f"\nTrainable parameters: " +
          ", ".join(f"{l}={r['params']}" for l, r in runs.items()))
    print(table("Training throughput", [
        ("`train_runtime` (s)", metric("train_runtime", "{:.1f}")),
        ("`train_samples_per_second`", metric("train_samples_per_second", "{:.3f}")),
        ("`train_steps_per_second`", metric("train_steps_per_second", "{:.3f}")),
        ("Avg step time from `[Comm]` (s)",
         lambda r: "{:.4f}".format(
             sum(c[1] for c in r["comm_steps"]) / len(r["comm_steps"])) if r["comm_steps"] else "n/a"),
    ], runs))
    print(table("Inference (evaluation) throughput", [
        ("`eval_samples_per_second`", metric("eval_samples_per_second", "{:.3f}")),
        ("`eval_runtime` (s)", metric("eval_runtime", "{:.3f}")),
        ("`eval_loss`", metric("eval_loss", "{:.4f}")),
    ], runs))
    print(table("Communication", [
        ("Total measured comm time (s)", summ("measured total comm time (whole run)")),
        ("Total measured comm bytes", summ("measured total bytes communicated")),
        ("Avg comm time per all-reduce (s)", summ("measured avg comm time / all-reduce call")),
        ("Avg comm time per optimizer step (s)", summ("measured avg comm time / optimizer step")),
        ("Theoretical grad payload per step", summ("theoretical grad payload / step (fp32)")),
        ("Theoretical ring bytes/GPU/step", summ("theoretical ring all-reduce bytes/GPU/step")),
    ], runs))


SYNTHETIC = """\
[NCCL] World size: 4
[Comm] Trainable parameters: 774,030,080
[Comm] Theoretical grad payload / step (fp32): 3,096,120,320 bytes
{'loss': 11.0, 'grad_norm': 1.0, 'learning_rate': 1e-06, 'epoch': 0.01}
[Comm] step=10 avg_step_time_s=0.5000 cumulative_comm_time_s=1.0000 cumulative_comm_bytes=1000
{'loss': 10.5, 'grad_norm': 1.0, 'learning_rate': 1e-06, 'epoch': 0.02}
[Comm] step=20 avg_step_time_s=0.5000 cumulative_comm_time_s=2.0000 cumulative_comm_bytes=2000
{'loss': 10.0, 'grad_norm': 1.0, 'learning_rate': 1e-06, 'epoch': 0.03}
[Comm] step=30 avg_step_time_s=1.0000 cumulative_comm_time_s=3.0000 cumulative_comm_bytes=3000
{'train_runtime': 20.0, 'train_samples_per_second': 60.0, 'train_steps_per_second': 1.5}
======================================================================
[Comm] Communication summary
[Comm]   world_size:                              4
[Comm]   trainable parameters:                    774,030,080
[Comm]   theoretical grad payload / step (fp32):   3,096,120,320 bytes
[Comm]   theoretical ring all-reduce bytes/GPU/step:4,644,180,480 bytes
[Comm]   measured total comm time (whole run):     3.0000 s
[Comm]   measured total bytes communicated:        3,000 bytes
[Comm]   measured avg comm time / all-reduce call: 0.001000 s
[Comm]   measured avg comm time / optimizer step:  0.100000 s
======================================================================
[Inference] Final evaluation (inference) performance
[Inference]   eval_loss: 9.5
[Inference]   eval_runtime: 4.0
[Inference]   eval_samples_per_second: 60.0
"""


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "fake.txt")
        p.write_text(SYNTHETIC, encoding="utf-8")
        r = parse(p)

    assert r["params"] == "774,030,080", r["params"]
    # steps inferred from logging_steps=10 since the dicts carry no 'step' key
    assert r["losses"] == [(10, 11.0), (20, 10.5), (30, 10.0)], r["losses"]
    # eval_loss must NOT be picked up as a training loss point
    assert all(l != 9.5 for _, l in r["losses"])
    # cumulative time: 10*0.5 + 10*0.5 + 10*1.0 = 20 s, matches train_runtime
    assert r["times"] == {10: 5.0, 20: 10.0, 30: 20.0}, r["times"]
    assert abs(r["total_time_s"] - 20.0) < 1e-9
    # [Inference] block wins over the periodic eval dicts
    assert r["metrics"]["eval_samples_per_second"] == 60.0
    assert r["metrics"]["eval_loss"] == 9.5
    assert r["metrics"]["train_runtime"] == 20.0
    assert r["summary"]["measured total comm time (whole run)"] == "3.0000"
    # this line has no space after the colon in train.py's f-string
    assert r["summary"]["theoretical ring all-reduce bytes/GPU/step"] == "4,644,180,480"
    assert loss_vs_time(r) == [(5.0, 11.0), (10.0, 10.5), (20.0, 10.0)], loss_vs_time(r)

    # a 1-GPU log has no [Comm] step lines at all -> fall back to train_runtime
    solo = parse_text_fallback()
    assert solo == [(10.0, 11.0), (20.0, 10.5)], solo
    print("selftest OK")


def parse_text_fallback():
    import tempfile
    text = ("{'loss': 11.0, 'epoch': 0.01}\n"
            "{'loss': 10.5, 'epoch': 0.02}\n"
            "{'train_runtime': 20.0}\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "f.txt")
        p.write_text(text, encoding="utf-8")
        return loss_vs_time(parse(p))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    runs = {"1-GPU": parse(sys.argv[1]), "4-GPU (DDP)": parse(sys.argv[2])}
    for f in plot(runs, "graphs"):
        print(f"wrote {f}")
    report(runs)
