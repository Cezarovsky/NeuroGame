"""
Encoding — state dict → spike rates (65 neuroni).

Structura vectorului de rates:
  [0:32]   — inel oponent: rata ∝ 1/distanță², pe bucket unghiular spre oponent
  [32:64]  — inel pereți: rata ∝ 1/distanță per direcție (N_WALL_DIRECTIONS raze)
  [64]     — stamina ratio [0, 1]

Rates sunt normalize în [0, 1] — LAVA spike generator le transformă în
frecvențe de spike în encoding.py → network.py.

Notă: "rate" nu înseamnă că folosim rate coding pur — e intrarea pentru
neuroni LIF. 1 neuron activ mai des = mai 1s per fereastra de 100 timestep.
"""
from __future__ import annotations

import math
import numpy as np
from game.arena import Arena, N_WALL_DIRECTIONS

N_OPPONENT_NEURONS = 32      # inel unghiular pentru oponent
N_WALL_NEURONS = N_WALL_DIRECTIONS  # == 32
N_STAMINA_NEURONS = 1
N_INPUT = N_OPPONENT_NEURONS + N_WALL_NEURONS + N_STAMINA_NEURONS  # = 65

# Parametri encoding oponent
DIST_MIN = 0.5               # distanță minimă (evităm div/0)
DIST_SCALE = 10.0            # normalizare distanță (rata maximă la ~1 unitate)


def _angle_bucket(angle_rad: float, n_buckets: int) -> int:
    """Convertește unghi [-π, π] în index bucket [0, n_buckets)."""
    normalized = angle_rad % (2 * math.pi)  # [0, 2π)
    return int(normalized / (2 * math.pi) * n_buckets) % n_buckets


def _opponent_ring(self_pos: np.ndarray, opp_pos: np.ndarray,
                   n_neurons: int = N_OPPONENT_NEURONS) -> np.ndarray:
    """
    Inel gaussian peste N_OPPONENT_NEURONS neuroni.

    Cel mai activ neuron e cel al cărui unghi corespunde direcției spre oponent.
    Activarea scade gaussian cu distanța unghiulară + scalată cu 1/dist².
    """
    delta = opp_pos - self_pos
    dist = max(float(np.linalg.norm(delta)), DIST_MIN)
    angle = math.atan2(delta[1], delta[0])

    # Amplitudine: mai aproape = activare mai puternică
    amplitude = min(1.0, (DIST_SCALE / dist) ** 2)

    # Sigma: ~2 bucket-uri (aproximativ 22.5°)
    sigma_buckets = 2.0

    rates = np.zeros(n_neurons)
    center = angle / (2 * math.pi) * n_neurons  # bucket-ul de centru (float)
    for i in range(n_neurons):
        # Distanța circulară în bucketi
        d = abs(((i - center + n_neurons / 2) % n_neurons) - n_neurons / 2)
        rates[i] = amplitude * math.exp(-0.5 * (d / sigma_buckets) ** 2)

    return rates


def _wall_ring(self_pos: np.ndarray, arena: Arena) -> np.ndarray:
    """
    Inel pereți: N_WALL_NEURONS valori, rata ∝ 1/dist (normalizat la 1).

    Neuronii mai activi = perete mai aproape în acea direcție.
    Raza maximă posibilă în arenă = diagonala.
    """
    dist_vec = arena.wall_distances_ring(self_pos)
    max_dist = math.hypot(arena.width, arena.height)
    # Proximitate: 0 = departe, 1 = lipit de perete
    proximity = 1.0 - np.clip(dist_vec / max_dist, 0.0, 1.0)
    return proximity


def encode_state(state_dict: dict) -> np.ndarray:
    """
    Transformă state_dict (primit din engine.py) în
    vectorul de rates shape (N_INPUT,) = (65,).

    state_dict conține:
      self_pos, opponent_pos, stamina, stamina_max, arena  (obligatoriu)
    """
    self_pos = state_dict["self_pos"]
    opponent_pos = state_dict["opponent_pos"]
    arena: Arena = state_dict["arena"]
    stamina = float(state_dict.get("stamina", 0.0))
    stamina_max = float(state_dict.get("stamina_max", 30.0))

    opp_ring = _opponent_ring(self_pos, opponent_pos)              # (32,)
    wall_ring = _wall_ring(self_pos, arena)                        # (32,)
    stamina_rate = np.array([stamina / max(stamina_max, 1e-9)])    # (1,)

    return np.concatenate([opp_ring, wall_ring, stamina_rate])     # (65,)


def rates_to_spikes(rates: np.ndarray, n_timesteps: int,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generează spike train binar shape (n_timesteps, N_INPUT) din vectorul de rates.

    rate ∈ [0, 1] = probabilitate de spike per timestep (Bernoulli independent).
    Folosit în modul standalone (fără LAVA).  Rețeaua LAVA are propriul
    spike generator — această funcție e pentru testare și pentru
    experiments/run.py cu backend NumPy.
    """
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random((n_timesteps, len(rates))) < rates).astype(np.float32)
