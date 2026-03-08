"""
Agenți: Prey (iepure) și Predator (lup).

Prey    — arc 180° perpendicular pe vectorul predator→prey (32 acțiuni)
Predator — 360° ponderat spre intercept (32 acțiuni)

Stamina: consumată per distanță parcursă, regenerare la turn%REGEN_EVERY==0.
Formula: distanța max per turn = stamina / log(turn + 2)
"""
from __future__ import annotations

import math
import numpy as np

N_ACTIONS = 32          # numărul de acțiuni (direcții discretizate)
REGEN_EVERY = 3         # stamina regenerează la fiecare 3 turnuri
STAMINA_REGEN = 5.0     # câtă stamina câștigă prada la regenerare
STAMINA_MAX = 30.0      # plafon stamina
SPEED_SCALE = 1.0       # factor global viteză (ajustabil pentru experimente)


class Prey:
    """Prada (iepure) — vrea să supraviețuiască."""

    def __init__(self, pos: np.ndarray, stamina: float = 20.0):
        self.pos = np.array(pos, dtype=float)
        self.stamina = float(stamina)
        self.prev_action: int = 0       # reținut; trimis la rețea ca context

    # ------------------------------------------------------------------
    # Stamina
    # ------------------------------------------------------------------

    def max_step(self, turn: int) -> float:
        """Distanța maximă pe care poate să o parcurgă în acest turn."""
        return self.stamina * SPEED_SCALE / math.log(turn + 2)

    def consume_stamina(self, distance: float) -> None:
        self.stamina = max(0.0, self.stamina - distance)

    def maybe_regen(self, turn: int) -> None:
        if turn % REGEN_EVERY == 0:
            self.stamina = min(STAMINA_MAX, self.stamina + STAMINA_REGEN)

    # ------------------------------------------------------------------
    # Spațiu de acțiuni
    # ------------------------------------------------------------------

    def action_positions(self, predator_pos: np.ndarray, turn: int) -> np.ndarray:
        """
        Returnează shape (N_ACTIONS, 2) — cele N_ACTIONS poziții candidate.

        Arc 180° centrat pe direcția de fugă (opusul vectorului pred→prey),
        ±90° față de aceasta.  Razele sunt egale cu max_step(turn).
        """
        flee_dir = self.pos - predator_pos
        norm = np.linalg.norm(flee_dir)
        if norm < 1e-9:
            base_angle = 0.0
        else:
            base_angle = math.atan2(flee_dir[1], flee_dir[0])

        step = self.max_step(turn)
        angles = base_angle + np.linspace(-math.pi / 2, math.pi / 2, N_ACTIONS)
        candidates = self.pos + step * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        return candidates

    def apply_action(self, action_idx: int, predator_pos: np.ndarray,
                     turn: int, arena) -> None:
        """Mută agentul la candidatul acțiunii, clampat în arenă."""
        candidates = self.action_positions(predator_pos, turn)
        new_pos = arena.clip_position(candidates[action_idx % N_ACTIONS])
        distance = float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos
        self.consume_stamina(distance)
        self.prev_action = action_idx
        self.maybe_regen(turn)


class Predator:
    """Prădătorul (lupul) — vrea să captureze prada."""

    def __init__(self, pos: np.ndarray, speed: float = 1.5):
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.prev_action: int = 0

    # ------------------------------------------------------------------
    # Intercept (nu pursuit)
    # ------------------------------------------------------------------

    def intercept_target(self, prey: "Prey", prey_velocity: np.ndarray,
                         n_turns: int = 2) -> np.ndarray:
        """
        Estimează unde va fi prada peste `n_turns` turnuri și returnează
        acel punct ca țintă de intercept.

        prey_velocity = deplasarea medie a prăzii din ultimele turnuri
        (calculată de engine și trimisă aici).
        """
        return prey.pos + prey_velocity * n_turns

    # ------------------------------------------------------------------
    # Spațiu de acțiuni
    # ------------------------------------------------------------------

    def action_positions(self, intercept_pt: np.ndarray) -> np.ndarray:
        """
        Returnează shape (N_ACTIONS, 2) — 360° uniform, raza = speed.

        Acțiunile aproape de direcția intercept primesc același tratament
        cu celelalte; rețeaua va fi recompensată să le aleagă (R-STDP).
        """
        angles = np.linspace(0.0, 2 * math.pi, N_ACTIONS, endpoint=False)
        candidates = self.pos + self.speed * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        return candidates

    def apply_action(self, action_idx: int, intercept_pt: np.ndarray,
                     arena) -> None:
        """Mută predatorul la candidatul ales, clampat în arenă."""
        candidates = self.action_positions(intercept_pt)
        new_pos = arena.clip_position(candidates[action_idx % N_ACTIONS])
        self.pos = new_pos
        self.prev_action = action_idx
