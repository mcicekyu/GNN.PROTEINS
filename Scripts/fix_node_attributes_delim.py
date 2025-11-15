import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "DATA" / "PROTEINS"
inp = ROOT / "PROTEINS_node_attributes.cleaned.txt"
out = ROOT / "PROTEINS_node_attributes.cleaned.fixed.txt"
bak = ROOT / "PROTEINS_node_attributes.cleaned.txt.bak"

if not inp.exists():
    raise FileNotFoundError(inp)

# backup original
if not bak.exists():
    inp.rename(bak)
    inp = bak
else:
    # if backup already exists, read from original path
    inp = ROOT / "PROTEINS_node_attributes.cleaned.txt.bak"

# regex for ints/floats with optional scientific notation
num_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

max_cols = 0
lines_in = 0
with inp.open('r') as fh_in, out.open('w') as fh_out:
    for line in fh_in:
        line = line.strip()
        if not line:
            continue
        lines_in += 1

        # Prefer splitting by comma (handles existing comma-separated rows).
        if ',' in line:
            parts = [p.strip() for p in line.split(',') if p.strip() != ""]
        else:
            # Fallback to numeric token regex for whitespace-separated or irregular rows
            parts = num_re.findall(line)

        if not parts:
            fh_out.write("\n")
            continue

        # join with comma + space as requested
        fh_out.write(", ".join(parts) + "\n")
        if len(parts) > max_cols:
            max_cols = len(parts)
           
print(f"Processed {lines_in} non-empty lines. Max columns found: {max_cols}")
print(f"Backup saved to: {inp}")
print(f"Fixed file written to: {out}")