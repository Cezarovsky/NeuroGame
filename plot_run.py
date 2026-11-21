"""
Evoluția unei rulări din jurnalul per episod (experiments/logs/run_*.csv).
Zero dependențe noi: scrie un HTML cu SVG, ca grafic_antrenament.html.

Folosire:
  ../.venv/bin/python plot_run.py                  # ultima rulare
  ../.venv/bin/python plot_run.py <fișier.csv>     # o rulare anume
  ../.venv/bin/python plot_run.py --window 100     # fereastra de medie mobilă

Output: HTML lângă CSV (același nume). Poate rula și în timpul antrenamentului —
jurnalul se scrie episod cu episod.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path

import numpy as np

LOG_DIR = Path("experiments/logs")

# Aceleași culori ca în vizualizator / grafic_antrenament.html
BG, PANEL, TEXT, DIM, GRID = "#0f0f1a", "#13131f", "#ccccdd", "#7a7a99", "#262638"
C_PRED, C_PREY, C_GREY = "#dc3c3c", "#50c850", "#8a96aa"

W, PANEL_W, PANEL_H, GAP, TOP = 1100, 500, 230, 40, 70


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: niciun episod încă")
    cols = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            cols[key] = np.array(vals, dtype=float)
        except ValueError:
            cols[key] = np.array(vals)
    return cols


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    """Medie mobilă trailing; primele w-1 puncte folosesc ce există până atunci."""
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(n - w, 0)
    return (c[n] - c[lo]) / (n - lo)


def panel(x0: int, y0: int, title: str, xs: np.ndarray,
          series: list[tuple[str, np.ndarray, str, str]], ymax: float | None = None) -> str:
    """Un grafic: titlu, axe cu 4 gradații, o polilinie per serie, legendă."""
    pl, pr, pt, pb = 50, 10, 24, 22            # padding interior
    iw, ih = PANEL_W - pl - pr, PANEL_H - pt - pb
    ymax = ymax if ymax is not None else max(float(np.nanmax(s[1])) for s in series) * 1.1 or 1.0
    xmin, xmax = float(xs.min()), float(max(xs.max(), xs.min() + 1))

    def px(x): return x0 + pl + (x - xmin) / (xmax - xmin) * iw
    def py(y): return y0 + pt + ih - min(max(y, 0.0), ymax) / ymax * ih

    out = [f'<rect x="{x0}" y="{y0}" width="{PANEL_W}" height="{PANEL_H}" fill="{PANEL}" rx="4"/>',
           f'<text x="{x0 + PANEL_W / 2}" y="{y0 + 15}" text-anchor="middle" font-size="11" fill="{TEXT}">{escape(title)}</text>']
    for i in range(5):
        y = ymax * i / 4
        out.append(f'<line x1="{x0 + pl}" x2="{x0 + pl + iw}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{x0 + pl - 5}" y="{py(y) + 3:.1f}" text-anchor="end" font-size="9" fill="{DIM}">{y:.3g}</text>')
    for x in np.linspace(xmin, xmax, 5):
        out.append(f'<text x="{px(x):.1f}" y="{y0 + PANEL_H - 6}" text-anchor="middle" font-size="9" fill="{DIM}">{x:.0f}</text>')
    lx, ly = x0 + pl + 8, y0 + pt + 10
    for label, ys, color, dash in series:
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(xs, ys))
        da = ' stroke-dasharray="5,4"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"{da}/>')
        out.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{ly - 3}" y2="{ly - 3}" stroke="{color}" stroke-width="2"{da}/>')
        out.append(f'<text x="{lx + 20}" y="{ly}" font-size="9" fill="{TEXT}">{escape(label)}</text>')
        lx += 20 + len(label) * 6 + 20      # monospace 9px ≈ 6px/caracter
    return "\n".join(out)


def render(path: Path, w: int) -> Path:
    d = load(path)
    ep, out = d["episode"], d["outcome"]
    r = lambda k: rolling(d[k], w)
    idle = (d["pred_dW_in"] == 0).mean() * 100

    panels = [
        ("Cine câștigă (% episoade)", [
            ("captură", rolling(out == "captured", w) * 100, C_PRED, ""),
            ("timeout — iepurele scapă", rolling(out == "timeout", w) * 100, C_PREY, ""),
            ("capcană", rolling(out == "trap", w) * 100, C_GREY, ""),
        ], 100.0),
        ("Durata episodului (tururi; 500 = timeout)", [("tururi", r("turns"), C_GREY, "")], 500.0),
        (f"Învățare ‖ΔW_in‖/episod — Fenrir inactiv în {idle:.0f}% din episoade", [
            ("Fenrir", r("pred_dW_in"), C_PRED, ""), ("Umbra", r("prey_dW_in"), C_PREY, ""),
        ], None),
        ("Altitudine (relevant din etapa 2)", [
            ("Fenrir medie", r("pred_alt_mean"), C_PRED, ""),
            ("Fenrir maxim", r("pred_alt_max"), C_PRED, "d"),
            ("Umbra medie", r("prey_alt_mean"), C_PREY, ""),
        ], None),
        ("Apropierea lup–iepure (unități)", [
            ("distanța minimă", r("dist_min"), C_PRED, ""),
            ("distanța la start", r("dist_start"), C_GREY, "d"),
        ], None),
        ("Iarba mâncată de Umbra (patch-uri/episod)", [("iarbă", r("grass_eaten"), C_PREY, "")], None),
    ]
    if "pred_sees_pct" in d:   # jurnale cu ceață + stamina lup
        panels += [
            ("Ceața: % din tururi în care își vede oponentul", [
                ("Fenrir o vede", r("pred_sees_pct"), C_PRED, ""),
                ("Umbra îl vede", r("prey_sees_pct"), C_PREY, ""),
            ], 100.0),
            ("Stamina lui Fenrir (max 30)", [
                ("medie", r("pred_stamina_mean"), C_PRED, ""),
                ("minimă", r("pred_stamina_min"), C_PRED, "d"),
            ], 30.0),
        ]

    h = TOP + (len(panels) + 1) // 2 * (PANEL_H + GAP)
    body = [f'<rect width="{W}" height="{h}" fill="{BG}"/>',
            f'<text x="{W / 2}" y="40" text-anchor="middle" font-size="15" font-weight="bold" fill="{TEXT}">'
            f'{escape(path.stem)}  |  {len(ep)} episoade  |  medie mobilă pe {w}</text>']
    for i, (title, series, ymax) in enumerate(panels):
        x0 = 40 + (i % 2) * (PANEL_W + 20)
        y0 = TOP + (i // 2) * (PANEL_H + GAP)
        body.append(panel(x0, y0, title, ep, series, ymax))

    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{escape(path.stem)}</title></head>"
            f"<body style='background:{BG};margin:0;padding:20px;'>"
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" style="font-family: monospace;">\n'
            + "\n".join(body) + "\n</svg></body></html>")
    out_path = path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", type=Path, help="jurnalul rulării (implicit: ultimul)")
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    path = args.csv or max(LOG_DIR.glob("run_*.csv"), key=lambda q: q.stat().st_mtime)
    print(render(path, args.window))
