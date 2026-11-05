"""Print and visualize every layer of the Reactnet_A_Qaw model.

The script reports both container modules and leaf layers. It also writes:

* ``reactnet_layers.txt``: readable layer table and hierarchy;
* ``reactnet_architecture.dot``: Graphviz representation of the module tree.

If Graphviz is installed and ``SAVE_PNG`` is enabled, a PNG is generated too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_A_Qaw import Reactnet

OUTPUT_DIR = ROOT / "cl" / "results" / "reactnet_inspection"
SAVE_PNG = True
MODEL_NUM_CLASSES = 100
MODEL_IMAGENET = False


def _is_binary(module: nn.Module, name: str) -> bool:
    return (
        "binary" in name.lower()
        or "bconv" in name.lower()
        or module.__class__.__name__.lower().startswith("hardbinary")
    )


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _shape_description(module: nn.Module) -> str:
    weights = [
        tuple(parameter.shape)
        for parameter in module.parameters(recurse=False)
    ]
    return ", ".join(str(shape) for shape in weights) if weights else "-"


def _module_rows(model: nn.Module) -> list[tuple[str, nn.Module, int]]:
    rows = []
    for name, module in model.named_modules():
        if not name:
            name = "<model>"
        rows.append((name, module, _parameter_count(module)))
    return rows


def _format_report(model: nn.Module) -> str:
    rows = _module_rows(model)
    lines = [
        "Reactnet_A_Qaw - complete module listing",
        "=" * 100,
        f"{'Name':55} {'Type':30} {'Role':10} {'Direct params':>14}  Parameters",
        "-" * 100,
    ]
    total = sum(parameter.numel() for parameter in model.parameters())
    for name, module, direct_params in rows:
        role = "BINARY" if _is_binary(module, name) else "non-binary"
        lines.append(
            f"{name:55.55} {module.__class__.__name__:30.30} "
            f"{role:10} {direct_params:14,d}  {_shape_description(module)}"
        )
    lines.extend(["-" * 100, f"Total parameters: {total:,}"])
    lines.append("")
    lines.append("Hierarchy")
    lines.append("=" * 100)
    for name, module, _ in rows:
        depth = 0 if name == "<model>" else name.count(".") + 1
        role = " [BINARY]" if _is_binary(module, name) else ""
        lines.append(f"{'  ' * depth}- {name}: {module.__class__.__name__}{role}")
    return "\n".join(lines) + "\n"


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_dot(model: nn.Module, output_path: Path) -> None:
    lines = [
        "digraph Reactnet {",
        '  graph [rankdir=LR, bgcolor="white", nodesep=0.25, ranksep=0.8];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];',
    ]
    node_ids: dict[str, str] = {}
    for index, (name, module, _) in enumerate(_module_rows(model)):
        key = "<model>" if name == "<model>" else name
        node_id = f"node_{index}"
        node_ids[key] = node_id
        role = "BINARY" if _is_binary(module, name) else "non-binary"
        color = "#f4a6a6" if role == "BINARY" else "#b8d8f0"
        label = f"{name}\\n{module.__class__.__name__}\\n{role}"
        lines.append(
            f'  {node_id} [label="{_dot_escape(label)}", fillcolor="{color}"];'
        )

    for name, _, _ in _module_rows(model):
        if name == "<model>":
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else "<model>"
        lines.append(f"  {node_ids[parent]} -> {node_ids[name]};")
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_reactnet(
    output_dir: str | Path = OUTPUT_DIR,
    *,
    save_png: bool = SAVE_PNG,
) -> tuple[Path, Path, Path | None]:
    """Print the model and save text, DOT, and optionally PNG representations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Reactnet(num_classes=MODEL_NUM_CLASSES, imagenet=MODEL_IMAGENET)
    report = _format_report(model)
    print(report, end="")

    text_path = output_dir / "reactnet_layers.txt"
    dot_path = output_dir / "reactnet_architecture.dot"
    text_path.write_text(report, encoding="utf-8")
    _write_dot(model, dot_path)

    png_path: Path | None = None
    if save_png:
        png_path = output_dir / "reactnet_architecture.png"
        try:
            subprocess.run(
                ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print(
                "\nGraphviz 'dot' non trovato: il file .dot è stato comunque creato."
            )
            png_path = None
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Graphviz non ha generato il PNG: {error.stderr}"
            ) from error

    print(f"\nReport salvato in: {text_path}")
    print(f"Diagramma Graphviz salvato in: {dot_path}")
    if png_path is not None:
        print(f"Immagine salvata in: {png_path}")
    return text_path, dot_path, png_path


if __name__ == "__main__":
    inspect_reactnet()
