"""
Terrain — relief cu dealuri, linie de vedere și sensing de pantă.

Altitudinea e o sumă de gaussiene (fiecare = un deal), deci e continuă și
derivabilă. Izoliniile sunt doar felul în care o desenăm, ca pe o hartă
topografică: o linie la fiecare CONTOUR_STEP unități de altitudine.

Ce dă terenul (compromisuri, nu doar avantaje):
  - de sus vezi mai departe (vision_radius crește cu altitudinea)
  - dealurile blochează linia de vedere (line_of_sight)
  - urcușul costă (climb_cost — de legat la stamina)

Coordonate: aceleași ca în Arena (origine stânga-jos, x dreapta, y sus).
"""
from __future__ import annotations

import math
import numpy as np

# Trebuie să fie același număr de direcții ca la pereți (encoding.py)
N_SLOPE_DIRECTIONS = 32

CONTOUR_STEP = 0.5     # altitudine între două izolinii
EYE_HEIGHT   = 0.3     # ochii agentului deasupra solului
BASE_VISION  = 8.0     # raza de vedere pe teren plat
VISION_GAIN  = 2.0     # +unități de rază pentru fiecare unitate de altitudine
LOS_STEP     = 0.25    # pasul de eșantionare pe linia de vedere
SLOPE_PROBE  = 1.5     # cât de departe „simte” panta în fiecare direcție
SLOPE_NORM   = 1.0     # diferența de altitudine care saturează un neuron
CLIMB_COST   = 5.0     # stamina per unitate de altitudine urcată (vârf de 4 ≈ 20 stamina)


class Terrain:
    def __init__(self, width: float, height: float, hills: list[tuple[float, float, float, float]] | None = None):
        self.width = float(width)
        self.height = float(height)
        # Fiecare deal: (center_x, center_y, altitudine_vârf, sigma)
        self.hills: list[tuple[float, float, float, float]] = hills or []

    @classmethod
    def random(cls, width: float, height: float, rng: np.random.Generator,
               n_hills: int = 4, h_range=(1.5, 4.0), sigma_range=(2.5, 5.0)) -> "Terrain":
        """Relief aleator. Dealurile sunt fixe pe sesiune, ca și capcanele."""
        hills = []
        for _ in range(n_hills):
            hills.append((rng.uniform(0, width), rng.uniform(0, height),
                          rng.uniform(*h_range), rng.uniform(*sigma_range)))
        return cls(width, height, hills)

    # ── altitudine ──────────────────────────────────────────────────────────

    def altitude(self, x, y):
        """Altitudinea în (x, y). Acceptă scalari sau array-uri numpy."""
        z = np.zeros(np.broadcast(x, y).shape)
        for cx, cy, h, s in self.hills:
            z = z + h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        return z if z.shape else float(z)

    def altitude_grid(self, resolution: float = 0.1) -> np.ndarray:
        """Grilă de altitudini [ny, nx], rândul 0 = y mic. Pentru randare."""
        xs = np.arange(0, self.width, resolution)
        ys = np.arange(0, self.height, resolution)
        gx, gy = np.meshgrid(xs, ys)
        return self.altitude(gx, gy)

    # ── vedere ──────────────────────────────────────────────────────────────

    def vision_radius(self, pos: np.ndarray, base: float = BASE_VISION) -> float:
        """Cine e sus vede mai departe. `base` = raza pe teren plat (diferă pe specie)."""
        return base + VISION_GAIN * self.altitude(pos[0], pos[1])

    def line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """True dacă din p1 se vede p2 (niciun deal nu iese deasupra razei)."""
        d = float(np.linalg.norm(p2 - p1))
        if d < 1e-6:
            return True
        z1 = self.altitude(p1[0], p1[1]) + EYE_HEIGHT
        z2 = self.altitude(p2[0], p2[1]) + EYE_HEIGHT
        n = max(2, int(d / LOS_STEP))
        t = np.linspace(0.0, 1.0, n)[1:-1]
        xs = p1[0] + t * (p2[0] - p1[0])
        ys = p1[1] + t * (p2[1] - p1[1])
        ray = z1 + t * (z2 - z1)
        return bool(np.all(self.altitude(xs, ys) <= ray))

    def can_see(self, observer: np.ndarray, target: np.ndarray, base: float = BASE_VISION) -> bool:
        """Ceața războiului: în rază ȘI cu linie de vedere liberă."""
        if np.linalg.norm(target - observer) > self.vision_radius(observer, base):
            return False
        return self.line_of_sight(observer, target)

    # ── sensing ─────────────────────────────────────────────────────────────

    def slope_ring(self, pos: np.ndarray) -> np.ndarray:
        """
        Panta pe N_SLOPE_DIRECTIONS direcții, codată pentru SNN în [0, 1].
        0.5 = plat, >0.5 = urcă în direcția aia, <0.5 = coboară.
        Nu spune „acolo e un deal” — doar ce simte sub picioare.
        """
        angles = np.linspace(0, 2 * math.pi, N_SLOPE_DIRECTIONS, endpoint=False)
        xs = pos[0] + SLOPE_PROBE * np.cos(angles)
        ys = pos[1] + SLOPE_PROBE * np.sin(angles)
        dz = self.altitude(xs, ys) - self.altitude(pos[0], pos[1])
        return np.clip(0.5 + 0.5 * dz / SLOPE_NORM, 0.0, 1.0)

    def climb_cost(self, p_from: np.ndarray, p_to: np.ndarray) -> float:
        """Cost suplimentar de stamina: doar urcușul costă, coborâșul nu."""
        dz = self.altitude(p_to[0], p_to[1]) - self.altitude(p_from[0], p_from[1])
        return CLIMB_COST * max(0.0, float(dz))

    def __repr__(self) -> str:
        return f"Terrain({self.width}x{self.height}, hills={len(self.hills)})"
