"""
Encoding — state dict → spike rates (105 neuroni).

Structura vectorului de rates:
  [0:32]   — inel oponent: rata ∝ 1/distanță², pe bucket unghiular spre oponent
             (zero când oponentul e ascuns de ceață)
  [32:64]  — inel pereți: rata ∝ 1/distanță per direcție (N_WALL_DIRECTIONS raze)
  [64]     — stamina ratio [0, 1]
  [65:73]  — inel iarbă: 8 neuroni direcționali spre cel mai apropiat patch
  [73:105] — inel pantă: >0.5 = urcă în direcția aia, <0.5 = coboară

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
N_GRASS_NEURONS = 8          # inel direcțional spre cel mai apropiat patch de iarbă
N_SLOPE_NEURONS = 32         # inel de pantă (terrain.slope_ring): 0.5 = plat
N_INPUT = (N_OPPONENT_NEURONS + N_WALL_NEURONS + N_STAMINA_NEURONS
           + N_GRASS_NEURONS + N_SLOPE_NEURONS)  # = 105

# Parametri encoding oponent
DIST_MIN = 0.5               # distanță minimă (evităm div/0)
DIST_SCALE = 10.0            # normalizare distanță (rata maximă la ~1 unitate)

# Parametri encoding iarbă
GRASS_DIST_SCALE = 5.0       # semnal maxim la 5 unități distanță
GRASS_SIGMA = 1.0            # lățimea gaussianei (bucket-uri)


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


def _grass_ring(self_pos: np.ndarray,
                patches: list[tuple[float, float]],
                n_neurons: int = N_GRASS_NEURONS) -> np.ndarray:
    """
    8 neuroni direcționali spre cel mai apropiat patch de iarbă.
    Activare 0 dacă nu există patch-uri.
    """
    if not patches:
        return np.zeros(n_neurons)

    # Găsim cel mai apropiat patch
    dists = [max(float(np.linalg.norm(self_pos - np.array(p))), DIST_MIN)
             for p in patches]
    nearest = np.array(patches[int(np.argmin(dists))])
    dist    = min(dists)

    delta     = nearest - self_pos
    angle     = math.atan2(delta[1], delta[0])
    amplitude = min(1.0, (GRASS_DIST_SCALE / dist) ** 2)

    rates  = np.zeros(n_neurons)
    center = angle / (2 * math.pi) * n_neurons
    for i in range(n_neurons):
        d = abs(((i - center + n_neurons / 2) % n_neurons) - n_neurons / 2)
        rates[i] = amplitude * math.exp(-0.5 * (d / GRASS_SIGMA) ** 2)

    return rates


def encode_state(state_dict: dict) -> np.ndarray:
    """
    Transformă state_dict (primit din engine.py) în
    vectorul de rates shape (N_INPUT,) = (105,).

    state_dict conține:
      self_pos, opponent_pos, stamina, stamina_max, arena  (obligatoriu)
      grass_patches: list[(x, y)] — opțional, patch-uri de iarbă active
      opponent_visible: bool — opțional (implicit True); False = inel oponent zero
      slope_ring: np.ndarray (32,) — opțional, panta pe direcții (implicit plat)
    """
    self_pos     = state_dict["self_pos"]
    opponent_pos = state_dict["opponent_pos"]
    arena: Arena = state_dict["arena"]
    stamina      = float(state_dict.get("stamina", 0.0))
    stamina_max  = float(state_dict.get("stamina_max", 30.0))
    grass        = state_dict.get("grass_patches", [])

    # Ceața războiului: oponentul nevăzut = inel tăcut (nu „știe” unde e)
    if state_dict.get("opponent_visible", True):
        opp_ring = _opponent_ring(self_pos, opponent_pos)           # (32,)
    else:
        opp_ring = np.zeros(N_OPPONENT_NEURONS)
    wall_ring    = _wall_ring(self_pos, arena)                      # (32,)
    stamina_rate = np.array([stamina / max(stamina_max, 1e-9)])     # (1,)
    grass_ring   = _grass_ring(self_pos, grass)                     # (8,)
    # Panta sub picioare; fără teren = plat peste tot
    slope_ring   = state_dict.get("slope_ring")
    if slope_ring is None:
        slope_ring = np.full(N_SLOPE_NEURONS, 0.5)                  # (32,)

    return np.concatenate([opp_ring, wall_ring, stamina_rate, grass_ring, slope_ring])  # (105,)


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
