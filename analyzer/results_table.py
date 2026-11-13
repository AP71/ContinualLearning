"""Summary table of every run in cl/results (one row per run, one column per statistic).

Writes to analyzer/comparison_plots/:
  - results_table.html : interactive table (click a header to sort ascending/descending,
                         text filter, "best run per method only")
  - results_table.csv  : same data, for Excel or LaTeX

Statistics (mean ± sample standard deviation over the available seeds):
  - Final Class-IL       : mean accuracy over the 10 tasks after the last task (all 100 logits)
  - Avg. incremental     : mean of the Class-IL accuracies after each task
  - Final Task-IL        : same with logits restricted to the task classes (indicator only)
  - Final BWT            : Class-IL backward transfer after the last task
  - Class-IL after task 1: accuracy after the first task (initial plasticity)
  - Total bit flips      : bits changed over the whole sequence, in millions and in % of binary bits
JOINT, linear probe and SLDA appear as reference rows (only the available columns).

Example:
    python -m analyzer.results_table
"""
from __future__ import annotations

import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cl.results_layout import RESULTS_ROOT, experiment_dirs  # noqa: E402

OUTPUT_DIR = ROOT / "analyzer" / "comparison_plots"

# (key, header, decimals, initial sort order: "desc" when higher is better)
COLUMNS = [
    ("cil_final", "Final Class-IL (%)", 2, "desc"),
    ("cil_avg", "Avg. incremental Class-IL (%)", 2, "desc"),
    ("til_final", "Final Task-IL (%)", 2, "desc"),
    ("bwt", "Final BWT", 2, "desc"),
    ("cil_task1", "Class-IL after task 1 (%)", 2, "desc"),
    ("flips_m", "Total bit flips (M)", 2, "asc"),
    ("flips_pct", "Bit flips (% of bits)", 2, "asc"),
]


# ----------------------------------------------------------------------------- data
def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _matrix(values: Any) -> np.ndarray | None:
    if not isinstance(values, list) or not values:
        return None
    return np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)


def _stat(values: list[float]) -> tuple[float, float] | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std


def _run_label(folder_name: str) -> str:
    text = folder_name.removeprefix("cifar100_avalanche_").removeprefix("cifar100_")
    for token in ("_checkpoint_resneta_3seeds", "_random_resneta_3seeds", "_resneta_3seeds"):
        if token in text:
            init = " (random)" if token.startswith("_random") else ""
            head, _, tail = text.partition(token)
            return (head + (" " + tail.strip("_").replace("_", " ") if tail.strip("_") else "") + init).strip()
    return text.replace("_", " ")


def _sequential_row(group: str, folder: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        return None
    cil_final, cil_avg, til_final, bwt, cil_task1, flips_m, flips_pct = ([] for _ in range(7))
    for run in seeds.values():
        cil = _matrix(run.get("standard_class_incremental") or run.get("r_matrix"))
        if cil is None:
            continue
        til = _matrix(run.get("task_incremental") or run.get("task_incremental_r_matrix"))
        cil_final.append(100 * np.nanmean(cil[-1]))
        cil_avg.append(100 * np.mean([np.nanmean(cil[t, : t + 1]) for t in range(len(cil))]))
        cil_task1.append(100 * cil[0, 0])
        if til is not None:
            til_final.append(100 * np.nanmean(til[-1]))
        if isinstance(run.get("bwt"), list) and run["bwt"]:
            bwt.append(float(run["bwt"][-1]))
        if isinstance(run.get("bit_flips"), list):
            total = float(sum(run["bit_flips"]))
            flips_m.append(total / 1e6)
            if run.get("total_binary_bits"):
                flips_pct.append(100 * total / float(run["total_binary_bits"]))
    if not cil_final:
        return None
    return {
        "group": group,
        "run": _run_label(folder.name),
        "folder": folder.name,
        "seeds": len(cil_final),
        "kind": "run",
        "cil_final": _stat(cil_final),
        "cil_avg": _stat(cil_avg),
        "til_final": _stat(til_final),
        "bwt": _stat(bwt),
        "cil_task1": _stat(cil_task1),
        "flips_m": _stat(flips_m),
        "flips_pct": _stat(flips_pct),
    }


def _reference_rows(folder: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    base = {"group": "Reference", "folder": folder.name, "kind": "reference"}
    experiment_type = str(payload.get("experiment_type", ""))
    seeds = payload.get("seeds") if isinstance(payload.get("seeds"), dict) else {}
    if experiment_type == "joint_training":
        runs = list(seeds.values())
        flips = [float(r.get("total_bit_flips") or sum(r.get("bit_flips") or [])) for r in runs]
        pct = [100 * f / r["total_binary_bits"] for f, r in zip(flips, runs) if r.get("total_binary_bits")]
        return [{**base, "run": _run_label(folder.name), "seeds": len(runs),
                 "cil_final": _stat([r.get("final_accuracy_percent") for r in runs]),
                 "flips_m": _stat([f / 1e6 for f in flips]), "flips_pct": _stat(pct)}]
    if experiment_type == "linear_probe_frozen_backbone":
        acc = payload.get("test_accuracy_at_best_val_percent", {})
        return [{**base, "run": "Linear probe (joint, frozen backbone)", "seeds": len(seeds),
                 "cil_final": (acc.get("mean"), acc.get("std", 0.0)) if acc.get("mean") is not None else None}]
    if experiment_type == "statistical_heads_frozen_backbone":
        slda = payload.get("heads", {}).get("slda", {})
        final, bwt = slda.get("final_accuracy_percent", {}), slda.get("bwt_percent", {})
        curve = slda.get("accuracy_vs_classes_percent") or []
        return [{**base, "run": "SLDA (frozen backbone, no buffer)", "seeds": len(seeds),
                 "cil_final": (final.get("mean"), final.get("std", 0.0)) if final.get("mean") is not None else None,
                 "cil_avg": (float(np.mean(curve)), 0.0) if curve else None,
                 "cil_task1": (float(curve[0]), 0.0) if curve else None,
                 "bwt": (bwt.get("mean"), bwt.get("std", 0.0)) if bwt.get("mean") is not None else None,
                 "flips_m": (0.0, 0.0), "flips_pct": (0.0, 0.0)}]
    return []


def collect_rows(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in experiment_dirs(results_root):
        payload = _read(folder / "results.json")
        if not payload:
            continue
        if str(payload.get("experiment_type", "")).lower() == "sequential":
            group = folder.parent.name if folder.parent != results_root else "(no folder)"
            row = _sequential_row(group, folder, payload)
            if row:
                rows.append(row)
        else:
            rows.extend(_reference_rows(folder, payload))
    # best run per method (final Class-IL)
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["kind"] == "run" and row.get("cil_final"):
            if row["group"] not in best or row["cil_final"][0] > best[row["group"]]["cil_final"][0]:
                best[row["group"]] = row
    for row in rows:
        row["best"] = row["kind"] == "run" and best.get(row["group"]) is row
    return rows


# ----------------------------------------------------------------------------- output
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        header = ["Method", "Run", "Seeds", "Best"]
        for key, title, _, _ in COLUMNS:
            header += [title, f"{title} std"]
        writer.writerow(header)
        for row in rows:
            line = [row["group"], row["run"], row["seeds"], "yes" if row.get("best") else ""]
            for key, _, decimals, _ in COLUMNS:
                value = row.get(key)
                line += [f"{value[0]:.{decimals}f}", f"{value[1]:.{decimals}f}"] if value else ["", ""]
            writer.writerow(line)


def _cell(value: tuple[float, float] | None, decimals: int, seeds: int) -> str:
    if not value or value[0] is None:
        return '<td class="num empty" data-v="">–</td>'
    mean, std = value
    std_html = f'<span class="std">± {std:.{decimals}f}</span>' if seeds > 1 else ""
    return f'<td class="num" data-v="{mean:.6f}">{mean:.{decimals}f}{std_html}</td>'


def write_html(rows: list[dict[str, Any]], path: Path) -> None:
    head_cells = ['<th data-type="text">Method</th>', '<th data-type="text">Run</th>',
                  '<th data-type="num" data-default="desc">Seeds</th>']
    head_cells += [f'<th data-type="num" data-default="{order}">{html.escape(title)}</th>'
                   for _, title, _, order in COLUMNS]
    body = []
    for row in rows:
        classes = [row["kind"]] + (["best"] if row.get("best") else [])
        star = '<span class="star" title="Best run of the method">★</span>' if row.get("best") else ""
        cells = [f'<td data-v="{html.escape(row["group"])}">{html.escape(row["group"])}</td>',
                 f'<td data-v="{html.escape(row["run"])}" title="{html.escape(row["folder"])}">'
                 f'{star}{html.escape(row["run"])}</td>',
                 f'<td class="num" data-v="{row["seeds"]}">{row["seeds"]}</td>']
        cells += [_cell(row.get(key), decimals, row["seeds"]) for key, _, decimals, _ in COLUMNS]
        body.append(f'<tr class="{" ".join(classes)}">{"".join(cells)}</tr>')

    page = TEMPLATE.replace("{{HEAD}}", "".join(head_cells)).replace("{{BODY}}", "\n".join(body))
    page = page.replace("{{DATE}}", f"{datetime.now():%Y-%m-%d %H:%M}").replace("{{N}}", str(len(rows)))
    path.write_text(page, encoding="utf-8")


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Split CIFAR-100 class-incremental results</title>
<style>
  :root { --bg:#fbfaf8; --fg:#1d1d1f; --muted:#6e6e73; --line:#e4e2dd; --accent:#2f5d8a; --best:#fff6dd; --ref:#f1f0ec; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161617; --fg:#ececec; --muted:#9a9aa0; --line:#2c2c2e; --accent:#7fb0e0; --best:#3a3220; --ref:#222224; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:32px 28px 48px; background:var(--bg); color:var(--fg);
         font:14px/1.45 -apple-system, "Segoe UI", Inter, Roboto, sans-serif; }
  h1 { font-size:20px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; font-size:13px; }
  .bar { display:flex; gap:18px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  .bar input[type=search] { padding:7px 10px; border:1px solid var(--line); border-radius:6px; min-width:260px;
         background:transparent; color:var(--fg); font:inherit; }
  .bar label { color:var(--muted); display:flex; gap:6px; align-items:center; cursor:pointer; user-select:none; }
  .wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { padding:8px 12px; border-bottom:1px solid var(--line); white-space:nowrap; text-align:left; }
  th { position:sticky; top:0; background:var(--bg); font-weight:600; font-size:12.5px; cursor:pointer; user-select:none; }
  th:hover { color:var(--accent); }
  th.sorted { color:var(--accent); }
  th .arrow { display:inline-block; width:12px; margin-left:4px; font-size:10px; }
  td.num, th:nth-child(n+3) { text-align:right; }
  .std { color:var(--muted); font-size:12px; margin-left:5px; }
  td.empty { color:var(--muted); }
  tr.best td { background:var(--best); }
  tr.reference td { background:var(--ref); font-style:italic; }
  .star { color:#c99a1e; margin-right:6px; font-style:normal; }
  tbody tr:hover td { filter:brightness(0.97); }
  .legend { color:var(--muted); font-size:12.5px; margin-top:14px; line-height:1.6; }
</style>
</head>
<body>
<h1>Split CIFAR-100, 10 tasks — method comparison</h1>
<p class="sub">{{N}} rows · generated on {{DATE}} · mean ± standard deviation over seeds · click a header to sort</p>
<div class="bar">
  <input type="search" id="filter" placeholder="Filter by method or run…">
  <label><input type="checkbox" id="bestOnly"> best run per method only</label>
  <label><input type="checkbox" id="showRef" checked> show references (JOINT, linear probe, SLDA)</label>
</div>
<div class="wrap">
<table id="t">
<thead><tr>{{HEAD}}</tr></thead>
<tbody>
{{BODY}}
</tbody>
</table>
</div>
<p class="legend">★ / highlighted row: best run of the method by final Class-IL. Italic rows: references, not continual learning methods.
Task-IL is reported as an indicator only. Bit flips (% of bits): bits changed over the whole sequence relative to the binary bits of the network.</p>
<script>
(function () {
  const table = document.getElementById("t");
  const tbody = table.tBodies[0];
  const headers = Array.from(table.tHead.rows[0].cells);
  const filter = document.getElementById("filter");
  const bestOnly = document.getElementById("bestOnly");
  const showRef = document.getElementById("showRef");
  let sortCol = 3, sortDir = "desc";

  headers.forEach((th, i) => {
    th.insertAdjacentHTML("beforeend", '<span class="arrow"></span>');
    th.addEventListener("click", () => {
      if (sortCol === i) sortDir = sortDir === "asc" ? "desc" : "asc";
      else { sortCol = i; sortDir = th.dataset.default || "asc"; }
      sort();
    });
  });

  function value(row, i) { return row.cells[i].dataset.v; }

  function sort() {
    const type = headers[sortCol].dataset.type;
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {
      const va = value(a, sortCol), vb = value(b, sortCol);
      if (va === "" && vb === "") return 0;
      if (va === "") return 1;            // missing values always last
      if (vb === "") return -1;
      const cmp = type === "num" ? parseFloat(va) - parseFloat(vb) : va.localeCompare(vb, "en");
      return sortDir === "asc" ? cmp : -cmp;
    });
    rows.forEach(r => tbody.appendChild(r));
    headers.forEach((th, i) => {
      th.classList.toggle("sorted", i === sortCol);
      th.querySelector(".arrow").textContent = i === sortCol ? (sortDir === "asc" ? "▲" : "▼") : "";
    });
  }

  function applyFilters() {
    const q = filter.value.trim().toLowerCase();
    Array.from(tbody.rows).forEach(r => {
      const text = (r.cells[0].textContent + " " + r.cells[1].textContent + " " + r.cells[1].title).toLowerCase();
      const isRef = r.classList.contains("reference");
      const visible = (!q || text.includes(q))
        && (!bestOnly.checked || r.classList.contains("best") || isRef)
        && (showRef.checked || !isRef);
      r.style.display = visible ? "" : "none";
    });
  }

  filter.addEventListener("input", applyFilters);
  bestOnly.addEventListener("change", applyFilters);
  showRef.addEventListener("change", applyFilters);
  sort();
})();
</script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if not rows:
        raise FileNotFoundError(f"No results found in {RESULTS_ROOT}")
    html_path, csv_path = OUTPUT_DIR / "results_table.html", OUTPUT_DIR / "results_table.csv"
    write_html(rows, html_path)
    write_csv(rows, csv_path)
    print(f"{len(rows)} rows\n  {html_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
