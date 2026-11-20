"""
Vizualizare Positional Encoding — Pygame, zero dependențe noi.
Output: positional_encoding.png

Doua panouri:
  1. Heatmap PE matrix (50 pozitii x 64 dimensiuni)
  2. Curbe sin/cos pentru cateva dimensiuni i
"""
import os, math
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

pygame.init()

W, H = 1200, 780
surf = pygame.Surface((W, H))

BG    = (15, 15, 26)
PANEL = (19, 19, 31)
GRID  = (30, 30, 51)
TEXT  = (204, 204, 221)
DIM   = (100, 100, 136)
WHITE = (255, 255, 255)

surf.fill(BG)
font_s = pygame.font.SysFont("monospace", 11)
font_m = pygame.font.SysFont("monospace", 13)
font_l = pygame.font.SysFont("monospace", 15, bold=True)

def draw_text(text, x, y, color=TEXT, font=None, anchor="left"):
    f = font or font_s
    r = f.render(str(text), True, color)
    if anchor == "center": x -= r.get_width() // 2
    elif anchor == "right": x -= r.get_width()
    surf.blit(r, (x, y))

# ── Compute PE matrix ──────────────────────────────────────────────
N_POS = 50    # pozitii (tokens)
D     = 64    # dimensiuni embedding

pe = [[0.0] * D for _ in range(N_POS)]
for pos in range(N_POS):
    for i in range(D // 2):
        denom = 10000 ** (2 * i / D)
        pe[pos][2 * i]     = math.sin(pos / denom)
        pe[pos][2 * i + 1] = math.cos(pos / denom)

# ── Global title ───────────────────────────────────────────────────
draw_text("Positional Encoding — Vizualizare",
          W // 2, 14, TEXT, font_l, anchor="center")
draw_text("PE(pos, 2i) = sin(pos / 10000^(2i/d))    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
          W // 2, 34, DIM, font_s, anchor="center")

# ══════════════════════════════════════════════════════════════════
# PANEL 1 — Heatmap (stanga)
# ══════════════════════════════════════════════════════════════════
HM_X, HM_Y = 40, 70
HM_W, HM_H = 560, 360

pygame.draw.rect(surf, PANEL, (HM_X, HM_Y, HM_W, HM_H), border_radius=4)
draw_text("Heatmap PE matrix  (50 pozitii × 64 dimensiuni)",
          HM_X + HM_W // 2, HM_Y + 8, TEXT, font_s, anchor="center")

PAD_L, PAD_T, PAD_R, PAD_B = 50, 28, 20, 30
cell_w = (HM_W - PAD_L - PAD_R) / D
cell_h = (HM_H - PAD_T - PAD_B) / N_POS

def val_to_color(v):
    # -1 → albastru inchis, 0 → gri, +1 → portocaliu
    if v >= 0:
        r = int(60  + v * (230 - 60))
        g = int(80  + v * (140 - 80))
        b = int(120 + v * (40  - 120))
    else:
        t = -v
        r = int(60  - t * (60  - 20))
        g = int(80  - t * (80  - 40))
        b = int(120 + t * (220 - 120))
    return (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))

for pos in range(N_POS):
    for dim in range(D):
        v = pe[pos][dim]
        color = val_to_color(v)
        rx = HM_X + PAD_L + dim * cell_w
        ry = HM_Y + PAD_T + pos * cell_h
        pygame.draw.rect(surf, color,
                         (int(rx), int(ry), max(1, int(cell_w)), max(1, int(cell_h))))

# Axe heatmap
for pos in range(0, N_POS + 1, 10):
    ry = HM_Y + PAD_T + pos * cell_h
    draw_text(str(pos), HM_X + PAD_L - 4, int(ry) - 5, DIM, anchor="right")
for dim in range(0, D + 1, 16):
    rx = HM_X + PAD_L + dim * cell_w
    draw_text(str(dim), int(rx), HM_Y + HM_H - PAD_B + 5, DIM, anchor="center")

draw_text("dim →", HM_X + PAD_L + D * cell_w / 2, HM_Y + HM_H - 14, DIM, anchor="center")
draw_text("pos", HM_X + 8, HM_Y + PAD_T + N_POS * cell_h / 2, DIM, anchor="center")

# Legenda colorbar
cb_x = HM_X + PAD_L
cb_y = HM_Y + HM_H - PAD_B + 16
cb_w = int(D * cell_w)
cb_h = 8
for xi in range(cb_w):
    v = -1.0 + 2.0 * xi / cb_w
    pygame.draw.line(surf, val_to_color(v), (cb_x + xi, cb_y), (cb_x + xi, cb_y + cb_h))
draw_text("-1", cb_x, cb_y + cb_h + 2, DIM)
draw_text("0",  cb_x + cb_w // 2, cb_y + cb_h + 2, DIM, anchor="center")
draw_text("+1", cb_x + cb_w, cb_y + cb_h + 2, DIM, anchor="right")

# Adnotare pattern vizibil
draw_text("← frecv. mare (i mic)   frecv. mica (i mare) →",
          HM_X + PAD_L + D * cell_w / 2, HM_Y + PAD_T - 14, (180, 180, 200),
          font_s, anchor="center")

# ══════════════════════════════════════════════════════════════════
# PANEL 2 — Curbe sin/cos (dreapta)
# ══════════════════════════════════════════════════════════════════
CV_X, CV_Y = 630, 70
CV_W, CV_H = 550, 360

pygame.draw.rect(surf, PANEL, (CV_X, CV_Y, CV_W, CV_H), border_radius=4)
draw_text("Curbe PE pentru diferite dimensiuni i  (d=64)",
          CV_X + CV_W // 2, CV_Y + 8, TEXT, font_s, anchor="center")

PL, PT, PR, PB = 55, 28, 15, 30
px0 = CV_X + PL
px1 = CV_X + CV_W - PR
py0 = CV_Y + PT
py1 = CV_Y + CV_H - PB

# Grid
for ytick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    py = int(py0 + (1.0 - (ytick + 1) / 2) * (py1 - py0))
    pygame.draw.line(surf, GRID, (px0, py), (px1, py), 1)
    draw_text(f"{ytick:.1f}", px0 - 4, py - 6, DIM, anchor="right")

for xtick in range(0, N_POS + 1, 10):
    px = int(px0 + xtick / (N_POS - 1) * (px1 - px0))
    pygame.draw.line(surf, GRID, (px, py0), (px, py1), 1)
    draw_text(str(xtick), px, py1 + 4, DIM, anchor="center")

draw_text("pozitie (token index) →", px0 + (px1-px0)//2, py1 + 16, DIM, anchor="center")
draw_text("valoare PE", px0 - 48, py0 + (py1-py0)//2, DIM, anchor="center")

# Curbe pentru i = 0, 1, 4, 16, 31
curves = [
    (0,  True,  (224,  82,  82), "i=0  sin (rapid)"),
    (0,  False, (224, 130,  82), "i=0  cos (rapid)"),
    (4,  True,  ( 82, 200,  82), "i=4  sin"),
    (16, True,  ( 82, 168, 224), "i=16 sin"),
    (31, True,  (180,  82, 224), "i=31 sin (lent)"),
]

for (i, use_sin, color, label) in curves:
    denom = 10000 ** (2 * i / D)
    points = []
    for pos in range(N_POS):
        v = math.sin(pos / denom) if use_sin else math.cos(pos / denom)
        px = int(px0 + pos / (N_POS - 1) * (px1 - px0))
        py = int(py0 + (1.0 - (v + 1) / 2) * (py1 - py0))
        points.append((px, py))
    if len(points) > 1:
        pygame.draw.lines(surf, color, False, points, 2)

# Legenda curbe
leg_x = px0 + 4
leg_y = py0 + 6
for (i, use_sin, color, label) in curves:
    pygame.draw.line(surf, color, (leg_x, leg_y + 5), (leg_x + 20, leg_y + 5), 2)
    draw_text(label, leg_x + 24, leg_y, TEXT)
    leg_y += 16

# ══════════════════════════════════════════════════════════════════
# PANEL 3 — Fingerprint doua pozitii (jos, lat)
# ══════════════════════════════════════════════════════════════════
FP_X, FP_Y = 40, 450
FP_W, FP_H = 1140, 290

pygame.draw.rect(surf, PANEL, (FP_X, FP_Y, FP_W, FP_H), border_radius=4)
draw_text("Fingerprint unic per pozitie — primele 32 dimensiuni  (pos=2 vs pos=20)",
          FP_X + FP_W // 2, FP_Y + 8, TEXT, font_s, anchor="center")

FPL, FPT, FPR, FPB = 55, 28, 20, 30
fpx0 = FP_X + FPL
fpx1 = FP_X + FP_W - FPR
fpy0 = FP_Y + FPT
fpy1 = FP_Y + FP_H - FPB

D_SHOW = 32
bar_w = (fpx1 - fpx0) / D_SHOW - 2

for ytick in [-1.0, 0.0, 1.0]:
    py = int(fpy0 + (1.0 - (ytick + 1) / 2) * (fpy1 - fpy0))
    pygame.draw.line(surf, GRID, (fpx0, py), (fpx1, py), 1)
    draw_text(f"{ytick:.0f}", fpx0 - 4, py - 6, DIM, anchor="right")

base_py = int(fpy0 + (fpy1 - fpy0) / 2)

for dim in range(D_SHOW):
    bx = fpx0 + dim * (bar_w + 2)

    # pos=2
    v2 = pe[2][dim]
    py2 = int(fpy0 + (1.0 - (v2 + 1) / 2) * (fpy1 - fpy0))
    top = min(py2, base_py)
    bh  = abs(base_py - py2)
    s = pygame.Surface((int(bar_w), bh), pygame.SRCALPHA)
    s.fill((224, 82, 82, 180))
    surf.blit(s, (int(bx), top))

    # pos=20
    v20 = pe[20][dim]
    py20 = int(fpy0 + (1.0 - (v20 + 1) / 2) * (fpy1 - fpy0))
    top20 = min(py20, base_py)
    bh20  = abs(base_py - py20)
    s2 = pygame.Surface((int(bar_w), bh20), pygame.SRCALPHA)
    s2.fill((82, 168, 224, 140))
    surf.blit(s2, (int(bx + bar_w * 0.5), top20))

    if dim % 4 == 0:
        draw_text(str(dim), int(bx + bar_w/2), fpy1 + 4, DIM, anchor="center")

# Legenda fingerprint
pygame.draw.rect(surf, (224, 82, 82), (fpx0 + 4, fpy0 + 6, 14, 10))
draw_text("pos=2", fpx0 + 22, fpy0 + 6, TEXT)
pygame.draw.rect(surf, (82, 168, 224), (fpx0 + 80, fpy0 + 6, 14, 10))
draw_text("pos=20  (pattern diferit → Attention le distinge)", fpx0 + 98, fpy0 + 6, TEXT)

draw_text("dim →", fpx0 + (fpx1-fpx0)//2, fpy1 + 16, DIM, anchor="center")

# Border panels
for rect in [(HM_X, HM_Y, HM_W, HM_H), (CV_X, CV_Y, CV_W, CV_H), (FP_X, FP_Y, FP_W, FP_H)]:
    pygame.draw.rect(surf, GRID, rect, width=1, border_radius=4)

# ── Save ───────────────────────────────────────────────────────────
out = "positional_encoding.png"
pygame.image.save(surf, out)
pygame.quit()
print(f"Salvat: {out}")
