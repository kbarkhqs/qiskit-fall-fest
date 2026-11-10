"""Task 5: inspect the saved, executed nmr.ipynb (no execution here).

Run from the repository root after
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace nmr.ipynb
with
    .venv/bin/python agents/agent2_implementation/check_executed_notebook.py
"""
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
nb = json.loads((REPO / "nmr.ipynb").read_text(encoding="utf-8"))

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
report = {
    "number_of_cells": len(nb["cells"]),
    "number_of_code_cells": len(code_cells),
    "execution_counts": [c["execution_count"] for _, c in code_cells],
    "kernelspec": nb["metadata"].get("kernelspec"),
    "language_version": nb["metadata"].get("language_info", {}).get("version"),
    "nbformat": [nb["nbformat"], nb["nbformat_minor"]],
    "errors": [],
    "stderr_outputs": [],
    "outputs_mentioning_warning": [],
    "images": [],
}
for i, c in code_cells:
    for o in c.get("outputs", []):
        if o["output_type"] == "error":
            report["errors"].append({"cell": i, "ename": o.get("ename"), "evalue": o.get("evalue")})
        text = "".join(o.get("text", [])) if o["output_type"] == "stream" else "".join(o.get("data", {}).get("text/plain", []))
        if o["output_type"] == "stream" and o.get("name") == "stderr":
            report["stderr_outputs"].append({"cell": i, "text": text[:300]})
        if "Warning" in text or "warning" in text:
            report["outputs_mentioning_warning"].append({"cell": i, "text": text[:300]})
        if "image/png" in o.get("data", {}):
            png = base64.b64decode(o["data"]["image/png"])
            name = f"notebook_cell_{i}_figure.png"
            (HERE / name).write_bytes(png)
            report["images"].append({"cell": i, "bytes": len(png), "file": name})

report["all_code_cells_executed_in_order"] = report["execution_counts"] == list(range(1, len(code_cells) + 1))
report["verification_cell_output"] = "".join(nb["cells"][15]["outputs"][0]["text"]) if nb["cells"][15].get("outputs") else None
report["last_code_cell_index"] = code_cells[-1][0]
report["last_code_cell_has_image"] = any(im["cell"] == code_cells[-1][0] for im in report["images"])
report["clean"] = (
    not report["errors"] and not report["stderr_outputs"] and not report["outputs_mentioning_warning"]
    and report["all_code_cells_executed_in_order"] and report["last_code_cell_has_image"]
)
(HERE / "check_executed_notebook.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
