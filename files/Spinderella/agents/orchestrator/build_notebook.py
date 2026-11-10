"""Insert the orchestrator's section 8/9 cells into nmr.ipynb (idempotent: cells tagged
metadata.added_by == "orchestrator" are removed first). Does not execute the notebook."""
import sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "orchestrator"))
from notebook_cells import SECTION8, SECTION9  # noqa: E402
from notebook_cells import SECTION1_VIS, SECTION10, SECTION11  # noqa: E402
try:
    from notebook_cells import SECTION9_J  # added once the J-coupling experiments exist
except ImportError:
    SECTION9_J = []

nb = nbformat.read(ROOT / "nmr.ipynb", as_version=4)
nb.cells = [c for c in nb.cells if c.metadata.get("added_by") != "orchestrator"]


def make(kind, src):
    cell = nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
    cell.metadata["added_by"] = "orchestrator"
    cell.pop("id", None)
    return cell


src = ["".join(c.source) for c in nb.cells]
i1 = next(i for i, s in enumerate(src) if 'print("Struqture Hamiltonian:' in s)
nb.cells[i1 + 1:i1 + 1] = [make(k, s) for k, s in SECTION1_VIS]
src = ["".join(c.source) for c in nb.cells]
i8 = next(i for i, s in enumerate(src) if s.startswith("## 8. The Matching Task"))
assert nb.cells[i8 + 1].cell_type == "code" and "plt.plot" in nb.cells[i8 + 1].source
nb.cells[i8 + 2:i8 + 2] = [make(k, s) for k, s in SECTION8]
src = ["".join(c.source) for c in nb.cells]
i9 = next(i for i, s in enumerate(src) if s.startswith("## 9. Explore the Noise Tolerance"))
nb.cells[i9 + 1:i9 + 1] = [make(k, s) for k, s in SECTION9 + SECTION9_J]
nb.cells += [make(k, s) for k, s in SECTION10 + SECTION11]
nbformat.validator.normalize(nb)
nbformat.write(nb, ROOT / "nmr.ipynb")
print(f"cells: {len(nb.cells)}; section 8 at {i8}, section 9 at {i9}; added {len(SECTION10) + len(SECTION1_VIS) + len(SECTION8) + len(SECTION9) + len(SECTION9_J)}")
