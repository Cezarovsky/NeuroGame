"""
Engine — bucla de turnuri secvențiale.

Ordinea per turn:
  1. prey primește state → rețea → acțiune → se mișcă
  2. predator primește state → rețea → acțiune → se mișcă
  3. verificare captură → terminare episod
  4. R-STDP update (reward cu fereastra de 3 turnuri înapoi)

Engine nu știe nimic despre LAVA — primește două funcții `policy_fn`:
    action_idx = policy_fn(state_dict) -> int

Astfel putem testa mecanica jocului cu politici random înainte să
conectăm rețelele neuromorphice.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from game.arena import Arena
from game.agents import Prey, Predator, N_ACTIONS

VELOCITY_WINDOW = 4          # câte turnuri ținute pentru estimare viteză pradă
CAPTURE_THRESHOLD = 1.5      # distanță (unități) la care se consideră captură
MAX_TURNS = 500              # episod limitat pentru antrenament

PolicyFn = Callable[[dict], int]


# ------------------------------------------------------------------
# Structuri de date
# ------------------------------------------------------------------

@dataclass
class StepResult:
    turn: int
    prey_pos: np.ndarray
    pred_pos: np.ndarray
    prey_action: int
    pred_action: int
    prey_reward: float
    pred_reward: float
    done: bool
    info: str = ""


@dataclass
class EpisodeStats:
    turns: int
    captured: bool
    prey_total_reward: float
    pred_total_reward: float
    prey_stamina_end: float


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class GameEngine:
    def __init__(
        self,
        arena: Arena,
        prey: Prey,
        predator: Predator,
        prey_policy: PolicyFn,
        pred_policy: PolicyFn,
        capture_threshold: float = CAPTURE_THRESHOLD,
        max_turns: int = MAX_TURNS,
    ):
        self.arena = arena
        self.prey = prey
        self.predator = predator
        self.prey_policy = prey_policy
        self.pred_policy = pred_policy
        self.capture_threshold = capture_threshold
        self.max_turns = max_turns

        # Istoric pozițe pradă (pentru estimare viteză)
        self._prey_history: deque[np.ndarray] = deque(
            [prey.pos.copy()], maxlen=VELOCITY_WINDOW + 1
        )
        self.turn: int = 0
        self.done: bool = False

        # Log complet de episod (opțional — folosit de experiments/run.py)
        self.history: List[StepResult] = []

    # ------------------------------------------------------------------
    # Estimare viteză pradă (pentru intercept)
    # ------------------------------------------------------------------

    def _prey_velocity(self) -> np.ndarray:
        """Viteza medie estimată din ultimele VELOCITY_WINDOW turnuri."""
        if len(self._prey_history) < 2:
            return np.zeros(2)
        diffs = [
            self._prey_history[i] - self._prey_history[i - 1]
            for i in range(1, len(self._prey_history))
        ]
        return np.mean(diffs, axis=0)

    # ------------------------------------------------------------------
    # Construire state dict (intrare pentru policy functions)
    # ------------------------------------------------------------------

    def _prey_state(self) -> dict:
        """State dict pentru prada — encoding.py îl transformă în spike train."""
        return {
            "self_pos": self.prey.pos.copy(),
            "opponent_pos": self.predator.pos.copy(),
            "stamina": self.prey.stamina,
            "stamina_max": 30.0,
            "turn": self.turn,
            "arena": self.arena,
            "role": "prey",
        }

    def _pred_state(self) -> dict:
        """State dict pentru prădător."""
        prey_vel = self._prey_velocity()
        intercept = self.predator.intercept_target(self.prey, prey_vel, n_turns=2)
        return {
            "self_pos": self.predator.pos.copy(),
            "opponent_pos": self.prey.pos.copy(),
            "intercept_pt": intercept,
            "turn": self.turn,
            "arena": self.arena,
            "role": "predator",
        }

    # ------------------------------------------------------------------
    # Recompense
    # ------------------------------------------------------------------

    def _compute_rewards(self, captured: bool, in_trap: bool = False) -> Tuple[float, float]:
        """
        Returnează (prey_reward, pred_reward).
        R-STDP are nevoie de semnal scalar per turn.
        """
        if in_trap:
            return -10.0, 0.0   # predatorul nu e recompensat pt capcane
        if captured:
            return -10.0, +10.0
        return +0.1, -0.05

    # ------------------------------------------------------------------
    # Un singur turn
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        if self.done:
            raise RuntimeError("Episodul s-a terminat. Apelați reset().")

        # 1. Prada se mișcă
        prey_state = self._prey_state()
        prey_action = self.prey_policy(prey_state)
        prey_vel_before = self._prey_velocity()
        self.prey.apply_action(
            prey_action,
            predator_pos=self.predator.pos,
            turn=self.turn,
            arena=self.arena,
        )
        self._prey_history.append(self.prey.pos.copy())

        # 2. Predatorul se mișcă
        pred_state = self._pred_state()
        pred_action = self.pred_policy(pred_state)
        intercept_pt = pred_state["intercept_pt"]
        self.predator.apply_action(pred_action, intercept_pt, arena=self.arena)

        # 3. Verificare capcană (prey moare singur — segment-based pentru salturi mari)
        prev_prey_pos = self._prey_history[-2] if len(self._prey_history) >= 2 else self.prey.pos
        in_trap = self.arena.check_trap_segment(prev_prey_pos, self.prey.pos)

        # 4. Verificare captură de predator
        captured = (not in_trap) and self.arena.check_capture(
            self.prey.pos, self.predator.pos, self.capture_threshold
        )

        self.turn += 1
        timeout = self.turn >= self.max_turns

        done = in_trap or captured or timeout
        prey_reward, pred_reward = self._compute_rewards(captured, in_trap)

        if in_trap:
            info = "trap"
        elif captured:
            info = "captured"
        elif timeout:
            info = "timeout"
        else:
            info = ""

        result = StepResult(
            turn=self.turn,
            prey_pos=self.prey.pos.copy(),
            pred_pos=self.predator.pos.copy(),
            prey_action=prey_action,
            pred_action=pred_action,
            prey_reward=prey_reward,
            pred_reward=pred_reward,
            done=done,
            info=info,
        )

        self.history.append(result)
        self.done = done
        return result

    # ------------------------------------------------------------------
    # Episod complet
    # ------------------------------------------------------------------

    def run_episode(self) -> EpisodeStats:
        """Rulează până la terminare, returnează statistici summary."""
        while not self.done:
            self.step()

        last = self.history[-1]
        total_prey_r = sum(r.prey_reward for r in self.history)
        total_pred_r = sum(r.pred_reward for r in self.history)

        return EpisodeStats(
            turns=self.turn,
            captured=last.info in ("captured", "trap"),  # mort = mort
            prey_total_reward=total_prey_r,
            pred_total_reward=total_pred_r,
            prey_stamina_end=self.prey.stamina,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        prey_pos: Optional[np.ndarray] = None,
        pred_pos: Optional[np.ndarray] = None,
    ) -> None:
        """Resetează episodul, opțional cu poziții inițiale noi."""
        if prey_pos is not None:
            self.prey.pos = np.array(prey_pos, dtype=float)
        if pred_pos is not None:
            self.predator.pos = np.array(pred_pos, dtype=float)
        self.prey.stamina = 20.0
        self.prey.prev_action = 0
        self.predator.prev_action = 0
        self._prey_history = deque(
            [self.prey.pos.copy()], maxlen=VELOCITY_WINDOW + 1
        )
        self.turn = 0
        self.done = False
        self.history = []
