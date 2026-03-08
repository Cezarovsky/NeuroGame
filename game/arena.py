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
            return 0.001  # direcție paralelă cu perete — nu se întâmplă cu 32 direcții

        return max(0.001, min(d for d in distances if d > 0))

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
