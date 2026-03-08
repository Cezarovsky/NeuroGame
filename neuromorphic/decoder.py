"""
Decoder — spike train output (n_timesteps, 32) → action index (0..31).

Strategie: Winner-Take-All pe ultimele WINDOW_SIZE timestep-uri.
Neuronul cu cel mai multe spike-uri în ultimele WINDOW_SIZE timesteps
e câștigătorul = acțiunea aleasă.

De ce ultimele timestep-uri, nu toate?
  - Rețeaua LIF are tranziente inițiale (stare 0 la start turn).
  - Ultimii timestep-uri = răspuns stabil după convergență.
  - WINDOW_SIZE = 20 din 100 = ultimii 20%.

Tie-breaking: dacă mai mulți neuroni au același număr de spike-uri,
se alege aleator dintre câștigători (nu primul!) — previne bias spre
acțiunile cu index mic.

Funcție auxiliară:
  softmax_decode — transformă spike counts în distribuție de probabilitate
  (pentru analiza comportamentului, nu pentru selecție acțiune în training).
"""
from __future__ import annotations

import numpy as np

WINDOW_SIZE = 20   # ultimii timestep-uri din cei 100 per turn


def decode_action(spikes_out: np.ndarray,
                  rng: np.random.Generator | None = None) -> int:
    """
    Winner-Take-All pe ultimele WINDOW_SIZE timestep-uri.

    Args:
        spikes_out: shape (n_timesteps, N_OUTPUT) — ieșirea din NeuroNet.run()
        rng: pentru tie-breaking aleator (opțional; dacă None folosit np.random)

    Returns:
        action_idx: int în [0, N_OUTPUT)
    """
    n_timesteps = spikes_out.shape[0]
    window_start = max(0, n_timesteps - WINDOW_SIZE)
    window = spikes_out[window_start:]            # (WINDOW_SIZE, N_OUTPUT)

    counts = window.sum(axis=0)                   # (N_OUTPUT,) — spike count per neuron

    max_count = counts.max()

    if max_count == 0:
        # Rețeaua silențioasă — acțiune aleatoare (exploration implicit)
        if rng is not None:
            return int(rng.integers(0, len(counts)))
        return int(np.random.randint(0, len(counts)))

    winners = np.where(counts == max_count)[0]

    if len(winners) == 1:
        return int(winners[0])

    # Tie-breaking aleator
    if rng is not None:
        return int(rng.choice(winners))
    return int(np.random.choice(winners))


def spike_counts(spikes_out: np.ndarray) -> np.ndarray:
    """Returnează spike counts pe fereastra WTA. Shape (N_OUTPUT,)."""
    n_timesteps = spikes_out.shape[0]
    window_start = max(0, n_timesteps - WINDOW_SIZE)
    return spikes_out[window_start:].sum(axis=0)


def softmax_decode(spikes_out: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Distribuție softmax peste spike counts — pentru analiză/vizualizare.
    Nu e folosit în training.

    Returns: shape (N_OUTPUT,) sumând la 1.
    """
    counts = spike_counts(spikes_out).astype(np.float64)
    scaled = counts / (temperature + 1e-9)
    scaled -= scaled.max()   # stabilitate numerică
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum()
