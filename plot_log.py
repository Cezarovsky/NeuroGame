"""
Grafic comportament Fenrir vs Umbra — Pygame, zero dependențe noi.
Output: grafic_antrenament.png
"""
import re
import math
import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"   # headless — fără fereastră
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

LOG_FILE = Path("antrenament.log")

# ── Parsare ────────────────────────────────────────────────────────
ep_re = re.compile(
    r"Ep\s+(\d+)\s+\|\s+captures\s+(\d+)\s+\([\d.]+%\)\s+\|\s+timeouts\s+(\d+)"
    r"\s+\|\s+ε=([\d.]+)"
)
cons_re = re.compile(
    r"\[CONSOLIDARE Ep\s+(\d+)\].*?ΔW_in=([\d.]+).*?Δcap=([\d.]+).*?tout/100=(\d+).*?cap%=([\d.]+)"
)

eps, captures_cum, timeouts_cum, epsilons = [], [], [], []
c_eps, dw_in, cap_pct, tout_win = [], [], [], []

with open(LOG_FILE) as f:
    for line in f:
        m = ep_re.search(line)
        if m:
            eps.append(int(m.group(1)))
            captures_cum.append(int(m.group(2)))
            timeouts_cum.append(int(m.group(3)))
            epsilons.append(float(m.group(4)))
            continue
        m = cons_re.search(line)
        if m:
            c_eps.append(int(m.group(1)))
            dw_in.append(float(m.group(2)))
            cap_pct.append(float(m.group(5)))
            tout_win.append(int(m.group(4)))

cap_rate_window, tout_delta = [], []
for i in range(len(eps)):
    prev_c = captures_cum[i-1] if i > 0 else 0
    prev_t = timeouts_cum[i-1] if i > 0 else 0
    cap_rate_window.append((captures_cum[i] - prev_c) / 50.0 * 100)
    tout_delta.append(timeouts_cum[i] - prev_t)

dw_pct = [v * 100 for v in dw_in]

# ── Pygame setup ───────────────────────────────────────────────────
pygame.init()
W, H = 1200, 860
surf = pygame.Surface((W, H))

BG     = (15, 15, 26)
PANEL  = (19, 19, 31)
GRID   = (30, 30, 51)
TEXT   = (204, 204, 221)
DIM    = (100, 100, 136)
FEN    = (224, 82, 82)
UMB    = (82, 168, 224)
GRN    = (126, 200, 126)
PRAG   = (255, 204, 68)
WHITE  = (255, 255, 255)

surf.fill(BG)
font_s = pygame.font.SysFont("monospace", 11)
font_m = pygame.font.SysFont("monospace", 13)
font_l = pygame.font.SysFont("monospace", 16, bold=True)

def draw_text(text, x, y, color=TEXT, font=None, anchor="left"):
    f = font or font_s
    rendered = f.render(str(text), True, color)
    if anchor == "center":
        x -= rendered.get_width() // 2
    elif anchor == "right":
        x -= rendered.get_width()
    surf.blit(rendered, (x, y))

def lerp(a, b, t): return a + (b - a) * t

def to_px(val, vmin, vmax, pmin, pmax):
    if vmax == vmin: return pmin
    return int(lerp(pmin, pmax, (val - vmin) / (vmax - vmin)))

def nice_ticks(vmin, vmax, n=5):
    rng = vmax - vmin
    if rng == 0: return [vmin]
    raw = rng / n
    mag = 10 ** math.floor(math.log10(max(abs(raw), 1e-9)))
    step = max(round(raw / mag) * mag, mag)
    first = math.ceil(vmin / step) * step
    ticks = []
    v = first
    while v <= vmax + step * 0.01:
        ticks.append(v)
        v += step
    return ticks

def draw_panel(rect, title, datasets, y_min, y_max, baselines=None):
    """
    datasets: list of dict {xs, ys, color, label, style: 'line'|'bar'|'dot'}
    baselines: list of (value, color, label)
    """
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, PANEL, rect, border_radius=4)
    pygame.draw.rect(surf, GRID,  rect, width=1, border_radius=4)

    draw_text(title, rx + rw//2, ry + 5, TEXT, font_s, anchor="center")

    PAD_L, PAD_R, PAD_T, PAD_B = 52, 12, 22, 22
    px0 = rx + PAD_L
    px1 = rx + rw - PAD_R
    py0 = ry + PAD_T
    py1 = ry + rh - PAD_B

    ep_min = eps[0] if eps else 0
    ep_max = eps[-1] if eps else 1

    # Y grid + labels
    for ytick in nice_ticks(y_min, y_max, 4):
        py = to_px(ytick, y_min, y_max, py1, py0)
        pygame.draw.line(surf, GRID, (px0, py), (px1, py), 1)
        label = f"{ytick:.0f}" if abs(ytick) >= 1 else f"{ytick:.2f}"
        draw_text(label, px0 - 4, py - 6, DIM, anchor="right")

    # X grid + labels
    for xtick in nice_ticks(ep_min, ep_max, 5):
        px = to_px(xtick, ep_min, ep_max, px0, px1)
        pygame.draw.line(surf, GRID, (px, py0), (px, py1), 1)
        draw_text(int(xtick), px, py1 + 4, DIM, anchor="center")

    # Baselines
    if baselines:
        for bval, bcol, blabel in baselines:
            bpy = to_px(bval, y_min, y_max, py1, py0)
            s = pygame.Surface((px1 - px0, 1), pygame.SRCALPHA)
            for i in range(0, px1 - px0, 7):
                pygame.draw.line(s, bcol, (i, 0), (min(i+4, px1-px0), 0), 1)
            surf.blit(s, (px0, bpy))
            draw_text(blabel, px1 - 2, bpy - 12, bcol, anchor="right")

    # Datasets
    legend_x = px0 + 4
    for ds in datasets:
        xs, ys, color, label, style = ds["xs"], ds["ys"], ds["color"], ds["label"], ds["style"]
        if not xs: continue

        points = [
            (to_px(x, ep_min, ep_max, px0, px1),
             to_px(y, y_min, y_max, py1, py0))
            for x, y in zip(xs, ys)
        ]

        base_py = to_px(0, y_min, y_max, py1, py0)

        if style in ("line", "area"):
            if style == "area" and len(points) > 1:
                poly = points + [(points[-1][0], base_py), (points[0][0], base_py)]
                fill_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
                pygame.draw.polygon(fill_surf, (*color, 40),
                    [(p[0]-rx, p[1]-ry) for p in poly])
                surf.blit(fill_surf, (rx, ry))
            if len(points) > 1:
                pygame.draw.lines(surf, color, False, points, 2)

        elif style == "bar":
            bar_w = max(2, (px1 - px0) // max(len(xs), 1) - 1)
            for (px, py) in points:
                top = min(py, base_py)
                bh  = abs(base_py - py)
                if bh > 0:
                    pygame.draw.rect(surf, color,
                        (px - bar_w//2, top, bar_w, bh))
                    # transparent overlay
                    s2 = pygame.Surface((bar_w, bh), pygame.SRCALPHA)
                    s2.fill((*color, 160))
                    surf.blit(s2, (px - bar_w//2, top))

        elif style == "dot":
            if len(points) > 1:
                pygame.draw.lines(surf, (*color, 120) if len(color)==3 else color,
                                  False, points, 1)
            for (px, py) in points:
                pygame.draw.circle(surf, color, (px, py), 4)
                pygame.draw.circle(surf, PANEL, (px, py), 2)

        # Legend item
        if label:
            pygame.draw.rect(surf, color, (legend_x, py0 + 4, 14, 3))
            draw_text(label, legend_x + 18, py0 + 1, TEXT)
            legend_x += len(label) * 7 + 28

    # Border
    pygame.draw.rect(surf, DIM, (px0, py0, px1-px0, py1-py0), width=1)


# ── Layout ─────────────────────────────────────────────────────────
MARGIN = (55, 50, 20, 20)   # L, T, R, B
GAP    = 16
T_H    = 30

ml, mt, mr, mb = MARGIN
pw2 = (W - ml - mr - GAP) // 2
ph  = (H - mt - mb - T_H - GAP * 2) // 3

# Global title
draw_text(f"NeuroGame — Fenrir vs Umbra  |  ep {eps[0]}–{eps[-1]}",
          W//2, mt + 8, TEXT, font_l, anchor="center")

# Panel 1 — capture rate (full width)
cr_min = max(0, min(cap_rate_window) - 2)
cr_max = max(cap_rate_window) + 4
draw_panel(
    (ml, mt + T_H, W - ml - mr, ph),
    "Fenrir — Rată captură per 50 ep (%)",
    [{"xs": eps, "ys": cap_rate_window, "color": FEN, "label": "captură %/50ep", "style": "area"}],
    cr_min, cr_max,
    baselines=[(20, PRAG, "baseline ~20%")]
)

# Panel 2 — timeouts cumulate
y2 = mt + T_H + ph + GAP
draw_panel(
    (ml, y2, pw2, ph),
    "Umbra — Timeouts cumulate",
    [{"xs": eps, "ys": timeouts_cum, "color": UMB, "label": "timeouts", "style": "area"}],
    0, max(timeouts_cum) + 1 if timeouts_cum else 5,
)

# Panel 3 — timeouts per 50ep
draw_panel(
    (ml + pw2 + GAP, y2, pw2, ph),
    "Umbra — Timeouts / 50 ep",
    [{"xs": eps, "ys": tout_delta, "color": UMB, "label": "", "style": "bar"}],
    0, max(tout_delta) + 1 if tout_delta else 3,
)

# Panel 4 — ΔW_in + cap% + tout_win (full width)
y3 = y2 + ph + GAP
all_y = dw_pct + cap_pct + [float(t) for t in tout_win]
y4_max = max(all_y) + 5 if all_y else 50
draw_panel(
    (ml, y3, W - ml - mr, ph),
    "SNN — ΔW_in×100 (verde)  cap%/100ep (roșu)  tout/100ep (albastru)",
    [
        {"xs": c_eps, "ys": dw_pct,                    "color": GRN, "label": "ΔW_in×100", "style": "dot"},
        {"xs": c_eps, "ys": cap_pct,                   "color": FEN, "label": "cap%",       "style": "dot"},
        {"xs": c_eps, "ys": [float(t) for t in tout_win], "color": UMB, "label": "tout/100ep","style": "bar"},
    ],
    0, y4_max,
    baselines=[
        (1.0,  PRAG, "ΔW_in prag (0.01)"),
        (10.0, UMB,  "tout min (10)"),
    ]
)

# ── Save ───────────────────────────────────────────────────────────
out = Path("grafic_antrenament.png")
pygame.image.save(surf, str(out))
pygame.quit()
print(f"Salvat: {out.resolve()}")
