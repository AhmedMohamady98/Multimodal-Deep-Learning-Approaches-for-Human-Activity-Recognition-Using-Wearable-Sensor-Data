"""Generate the 7 per-method analysis notebooks from `_per_method_template.py`.

Each output is a real .py file (jupytext-paired with .ipynb after conversion):
    03_late_fusion_analysis.py
    04_gmf_analysis.py
    05_mbt_analysis.py
    06_cma_analysis.py
    07_lmf_analysis.py
    08_decision_fusion_analysis.py
    09_cls_transformer_analysis.py

Run: PYTHONPATH=. python notebooks/v3/_build_per_method.py
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = (HERE / "_per_method_template.py").read_text()

# (file number, display name, method key)
METHODS = [
    ("03", "Late Fusion",     "late_fusion"),
    ("04", "GMF",             "gmf"),
    ("05", "MBT",             "mbt"),
    ("06", "CMA",             "cma"),
    ("07", "LMF",             "lmf"),
    ("08", "Decision Fusion", "decision_fusion"),
    ("09", "CLS Transformer", "cls_transformer"),
]

for num, display, key in METHODS:
    safe_key = key.replace("_", "_")
    out_name = f"{num}_{key}_analysis.py"
    content = (TEMPLATE
               .replace("{DISPLAY_NAME}", display)
               .replace("{METHOD_KEY}", key))
    (HERE / out_name).write_text(content)
    print(f"  wrote {out_name}")

# Convert each .py to a paired .ipynb (jupytext)
for num, _, key in METHODS:
    fname = f"{num}_{key}_analysis.py"
    subprocess.run(["jupytext", "--to", "ipynb", str(HERE / fname)], check=True)
print("\nDone — all 7 per-method notebooks generated + paired.")
