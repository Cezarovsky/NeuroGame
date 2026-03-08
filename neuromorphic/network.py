"""
Rețeaua neuromorphică: 65 → 256 (LIF + recurrent sparse 20%) → 32

Arhitectură:
  - Layer intrare: 65 neuroni LIF (rate-coded input)
  - Layer ascuns:  256 neuroni LIF cu recurrente sparse (memorie temporală)
  - Layer ieșire:  32 neuroni LIF (winner-take-all în decoder.py)

Backend: LAVA (lava-nc).  Dacă LAVA nu e instalat, există un fallback
NumPy (LIFNumpy) pentru testare fără hardware Loihi.

Recurrente: matrice W_rec shape (256, 256), sparse 20%
(20% din conexiuni sunt nenule la inițializare).
R-STDP aplică delta-W pe W_in și W_rec (nu pe W_out — dezbatere în
literatura: lăsăm W_out fix ca la Rao&Ballard 1999).
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

# Parametri LIF (unitiless — LAVA timestep)
TAU_MEM = 10            # constanta de timp membrană (timesteps)
V_THRESH = 1.0          # prag de spike
V_RESET = 0.0           # potential după spike
RECURRENT_SPARSITY = 0.20   # fracție conexiuni nenule în W_rec

# Dimensiuni rețea
N_INPUT = 65
N_HIDDEN = 256
N_OUTPUT = 32

# Seed reproductibil (schimbat din config în experiments/run.py)
DEFAULT_SEED = 42


# ==================================================================
# Fallback NumPy LIF — folosit când LAVA nu e disponibil
# ==================================================================

class LIFNumpy:
    """
    Simulare simplă LIF (Leaky Integrate-and-Fire) în NumPy.
    Folosită pentru testare locală pe macOS fără LAVA instalat.

    Ecuație membrană (Euler discretizat):
        v[t] = v[t-1] * (1 - 1/tau) + I[t]
        spike dacă v >= v_thresh → v = v_reset
    """

    def __init__(self, n_neurons: int, tau: float = TAU_MEM,
                 v_thresh: float = V_THRESH, v_reset: float = V_RESET):
        self.n = n_neurons
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = np.zeros(n_neurons, dtype=np.float32)

    def step(self, current: np.ndarray) -> np.ndarray:
        """Un timestep. Returnează spike binary (0/1) shape (n,)."""
        self.v = self.v * (1.0 - 1.0 / self.tau) + current
        spike = (self.v >= self.v_thresh).astype(np.float32)
        self.v[spike.astype(bool)] = self.v_reset
        return spike

    def reset(self) -> None:
        self.v[:] = 0.0


# ==================================================================
# Rețeaua principală
# ==================================================================

class NeuroNet:
    """
    Rețea 65 → 256 → 32 cu recurrente sparse în layerul ascuns.

    Interfață:
        spikes_out = net.run(input_rates, n_timesteps)   → (n_timesteps, 32)
        net.reset()                                       → resetează stările
        net.W_in, net.W_rec, net.W_out                   → accesibile pentru R-STDP
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        rng = np.random.default_rng(seed)
        self._init_weights(rng)
        self._init_neurons()

    # ------------------------------------------------------------------
    # Inițializare
    # ------------------------------------------------------------------

    def _init_weights(self, rng: np.random.Generator) -> None:
        """
        Xavier uniform pentru W_in și W_out.
        W_rec: sparse cu 20% conexiuni, magnitudine mică (stabilitate).
        """
        # W_in: (N_HIDDEN, N_INPUT)
        limit_in = np.sqrt(6.0 / (N_INPUT + N_HIDDEN))
        self.W_in = rng.uniform(-limit_in, limit_in,
                                (N_HIDDEN, N_INPUT)).astype(np.float32)

        # W_rec: (N_HIDDEN, N_HIDDEN), sparse 20%
        self.W_rec = np.zeros((N_HIDDEN, N_HIDDEN), dtype=np.float32)
        mask = rng.random((N_HIDDEN, N_HIDDEN)) < RECURRENT_SPARSITY
        np.fill_diagonal(mask, False)  # fără autoconexiuni
        limit_rec = np.sqrt(6.0 / (N_HIDDEN + N_HIDDEN)) * 0.5
        self.W_rec[mask] = rng.uniform(-limit_rec, limit_rec,
                                       mask.sum()).astype(np.float32)
        self.W_rec_mask = mask  # reținut pentru R-STDP (actualizăm doar conexiunile existente)

        # W_out: (N_OUTPUT, N_HIDDEN)
        limit_out = np.sqrt(6.0 / (N_HIDDEN + N_OUTPUT))
        self.W_out = rng.uniform(-limit_out, limit_out,
                                 (N_OUTPUT, N_HIDDEN)).astype(np.float32)

    def _init_neurons(self) -> None:
        self._hidden = LIFNumpy(N_HIDDEN)
        self._output = LIFNumpy(N_OUTPUT)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def run(self, input_rates: np.ndarray, n_timesteps: int,
            rng: np.random.Generator | None = None) -> Tuple[np.ndarray, dict]:
        """
        Rulează rețeaua pentru `n_timesteps` timestep-uri.

        Args:
            input_rates: shape (N_INPUT,) — ratele de spike [0, 1]
            n_timesteps: câte timestep-uri (default 100 per turn)
            rng: generator aleator (Bernoulli pentru spike input)

        Returns:
            spikes_out: shape (n_timesteps, N_OUTPUT)
            traces: dict cu spike-urile interne (pentru R-STDP)
        """
        if rng is None:
            rng = np.random.default_rng()

        spikes_hidden = np.zeros((n_timesteps, N_HIDDEN), dtype=np.float32)
        spikes_out    = np.zeros((n_timesteps, N_OUTPUT), dtype=np.float32)
        prev_hidden_spikes = np.zeros(N_HIDDEN, dtype=np.float32)

        for t in range(n_timesteps):
            # Generare spike input (Bernoulli)
            x = (rng.random(N_INPUT) < input_rates).astype(np.float32)

            # Current hidden = feed-forward + recurrent
            I_hidden = self.W_in @ x + self.W_rec @ prev_hidden_spikes
            h = self._hidden.step(I_hidden)
            spikes_hidden[t] = h
            prev_hidden_spikes = h

            # Current output
            I_out = self.W_out @ h
            o = self._output.step(I_out)
            spikes_out[t] = o

        traces = {
            "spikes_hidden": spikes_hidden,  # (T, N_HIDDEN)
            "spikes_out": spikes_out,         # (T, N_OUTPUT)
        }
        return spikes_out, traces

    # ------------------------------------------------------------------
    # Reset (între turnuri / episoade)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Resetează potențialele membrană (stare rețea) între episoade."""
        self._hidden.reset()
        self._output.reset()

    def reset_turn(self) -> None:
        """
        Resetează parțial între turnuri.
        Recurrentele acumulează informație temporală ÎN cadrul unui turn
        (100 timestep-uri). Între turnuri, opțional resetăm sau nu.
        Default: nu resetăm (recurrentele transmit context inter-turn).
        """
        pass  # Alegere explicită: starea persistă între turnuri în cadrul episodului

    # ------------------------------------------------------------------
    # Acces greutăți (pentru R-STDP)
    # ------------------------------------------------------------------

    def apply_weight_delta(self, dW_in: np.ndarray, dW_rec: np.ndarray) -> None:
        """
        Aplică actualizările de greutăți din R-STDP.
        W_rec e actualizat DOAR pe conexiunile existente (mask).
        """
        self.W_in += dW_in
        self.W_rec += dW_rec * self.W_rec_mask  # sparse mask păstrat
