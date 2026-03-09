"""
Arena — dreptunghi cu pereți.

Coordonate: origine stânga-jos, x creșe spre dreapta, y în sus.
Unitățile sunt abstracte (1.0 = 1 celulă).
"""
from __future__ import annotations

import math
import numpy as np

# Numărul de direcții pentru sensing pereți (trebuie să fie același în encoding.py)
N_WALL_DIRECTIONS = 32


class Arena:
    def __init__(self, width: float = 50.0, height: float = 50.0):
        self.width = float(width)
        self.height = float(height)
        # Liste de capcane: fiecare e (center_x, center_y, radius)
        self.traps: list[tuple[float, float, float]] = []

    def spawn_traps(self, n: int, rng: np.random.Generator,
                    radius: float = 4.0, margin: float = 0.0) -> None:
        """Plasează n capcane aleator, cu margin față de pereți."""
        self.traps = []
        for _ in range(n):
            cx = rng.uniform(margin, self.width  - margin)
            cy = rng.uniform(margin, self.height - margin)
            self.traps.append((cx, cy, radius))

    def check_trap(self, pos: np.ndarray) -> bool:
        """True dacă pos se află în interiorul oricărei capcane."""
        x, y = float(pos[0]), float(pos[1])
        for cx, cy, r in self.traps:
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return True
        return False

    def check_trap_segment(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        True dacă segmentul p1→p2 intersectează orice capcapănă.
        Folosit pentru a detecta traversarea capcanelor în salturi mari.
        """
        for cx, cy, r in self.traps:
            center = np.array([cx, cy])
            d = p2 - p1
            f = p1 - center
            a = float(np.dot(d, d))
            if a < 1e-12:
                return self.check_trap(p1)
            b = 2.0 * float(np.dot(f, d))
            c = float(np.dot(f, f)) - r * r
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                t1 = (-b - sq) / (2 * a)
                t2 = (-b + sq) / (2 * a)
                if (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2):
                    return True
        return False

    # ------------------------------------------------------------------
    # Poziție validă
    # ------------------------------------------------------------------

    def clip_position(self, pos: np.ndarray) -> np.ndarray:
        """Clampează poziția în interiorul arenei (fără să treacă prin perete)."""
        return np.clip(pos, [0.0, 0.0], [self.width, self.height])

    def is_inside(self, pos: np.ndarray) -> bool:
        return (0.0 <= pos[0] <= self.width) and (0.0 <= pos[1] <= self.height)

    # ------------------------------------------------------------------
    # Coliziune între agenți
    # ------------------------------------------------------------------

    def check_capture(self, prey_pos: np.ndarray, pred_pos: np.ndarray,
                      threshold: float = 1.0) -> bool:
        """True dacă distanța dintre agenți < threshold (captură)."""
        return float(np.linalg.norm(prey_pos - pred_pos)) < threshold

    # ------------------------------------------------------------------
    # Distanță la pereți (folosit de encoder)
    # ------------------------------------------------------------------

    def distance_to_wall(self, pos: np.ndarray, angle_rad: float) -> float:
        """
        Distanța de la `pos` la primul perete în direcția `angle_rad`.
        Calculul e ray-AABB simplu (cele 4 pereți ai dreptunghiului).

        Returnează distanța (>0) sau 0.001 dacă agentul e exact pe perete.
        """
        x, y = float(pos[0]), float(pos[1])
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        distances = []

        # Perete dreapta (x = width)
        if dx > 1e-9:
            distances.append((self.width - x) / dx)
        # Perete stânga (x = 0)
        if dx < -1e-9:
            distances.append(-x / dx)
        # Perete sus (y = height)
        if dy > 1e-9:
            distances.append((self.height - y) / dy)
        # Perete jos (y = 0)
        if dy < -1e-9:
            distances.append(-y / dy)

        if not distances:
            return 0.001

        positive = [d for d in distances if d > 0]
        if not positive:
            return 0.001

        return max(0.001, min(positive))

    def wall_distances_ring(self, pos: np.ndarray) -> np.ndarray:
        """
        Vector shape (N_WALL_DIRECTIONS,) cu distanțele la pereți pe fiecare
        direcție a inelului unghiular (0..2π împărțit în N_WALL_DIRECTIONS buckets).
        """
        angles = np.linspace(0.0, 2 * math.pi, N_WALL_DIRECTIONS, endpoint=False)
        return np.array([self.distance_to_wall(pos, a) for a in angles])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Arena(width={self.width}, height={self.height})"
