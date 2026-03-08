"""
R-STDP — Reward-modulated Spike-Timing Dependent Plasticity.

Implementare:
  1. Per fiecare timestep: eligibility trace actualizată cu STDP local.
  2. La sfârșitul unui turn (sau fereastra de reward): greutățile sunt
     actualizate cu reward × eligibility_trace_medie.

Regulă STDP asimetrică standard:
  Pre înainte de post (LTP): Δw = +A_plus  * exp(-Δt / τ_plus)
  Post înainte de pre (LTD): Δw = -A_minus * exp(-Δt / τ_minus)

Trace-urile pre și post sunt acumulate per timestep:
  x_pre[t]  += spike_pre[t];   x_pre  decay cu τ_plus
  x_post[t] += spike_post[t];  x_post decay cu τ_minus

Eligibility trace (per greutate W_ij):
  e[i,j] += x_pre[j] * spike_post[i] - x_post[i] * spike_pre[j]
  (LTP când pre activat înainte de post, LTD invers)

La reward:
  ΔW = lr * reward * mean(eligibility trace pe fereastra de 3 turnuri)

Fereastra de 3 turnuri: traces din ultimele 3 turnuri sunt stocate și
mediate cu exp decay (turnuri mai vechi = greutate mai mică).

Referință: Izhikevich (2007) "Solving the distal reward problem through
linkage of STDP and dopamine signaling."
"""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np

# Parametri STDP
TAU_PRE   = 20.0    # constanta de timp trace pre-sinaptic (timesteps)
TAU_POST  = 20.0    # constanta de timp trace post-sinaptic
A_PLUS    = 0.005   # LTP amplitude
A_MINUS   = 0.005   # LTD amplitude (simetric)

# Reward window
REWARD_WINDOW = 3      # numărul de turnuri ținute în fereastra de reward
REWARD_DECAY  = 0.7    # discount exponențial per turn în fereastră
LR            = 0.01   # learning rate final

# Clipping greutăți (stabilitate)
W_MAX =  1.0
W_MIN = -1.0


class RSTDPTracer:
    """
    Menține trace-urile STDP per turn și calculează delta-W la reward.

    Folosit separat pentru W_in (65×256) și W_rec (256×256).
    Un RSTDPTracer per pereche (pre_layer, post_layer).
    """

    def __init__(self, n_pre: int, n_post: int):
        self.n_pre = n_pre
        self.n_post = n_post

        # Trace-uri STDP (acumulate per timestep)
        self._x_pre  = np.zeros(n_pre,  dtype=np.float32)   # trace pre
        self._x_post = np.zeros(n_post, dtype=np.float32)   # trace post

        # Eligibility trace per turn (media pe 100 timestep-uri)
        # Fereastră REWARD_WINDOW turnuri
        self._elig_window: deque[np.ndarray] = deque(maxlen=REWARD_WINDOW)

        # Eligibility trace acumulată pe turn curent
        self._elig_turn = np.zeros((n_post, n_pre), dtype=np.float32)
        self._timestep_count = 0

    # ------------------------------------------------------------------
    # Actualizare per timestep
    # ------------------------------------------------------------------

    def update_timestep(self, spike_pre: np.ndarray,
                        spike_post: np.ndarray) -> None:
        """
        Apelat pentru fiecare timestep în interiorul unui turn.

        spike_pre:  shape (n_pre,)  — spikes strat anterior
        spike_post: shape (n_post,) — spikes strat următor
        """
        # Decay trace-uri
        self._x_pre  *= (1.0 - 1.0 / TAU_PRE)
        self._x_post *= (1.0 - 1.0 / TAU_POST)

        # Acumulare spike
        self._x_pre  += spike_pre
        self._x_post += spike_post

        # Eligibility trace:
        # LTP: post spike × pre trace  →  outer product
        # LTD: pre spike × post trace  →  outer product transposed
        ltp = np.outer(spike_post, self._x_pre)   # (n_post, n_pre)
        ltd = np.outer(self._x_post, spike_pre)   # (n_post, n_pre)

        self._elig_turn += A_PLUS * ltp - A_MINUS * ltd
        self._timestep_count += 1

    # ------------------------------------------------------------------
    # La finalul unui turn
    # ------------------------------------------------------------------

    def end_turn(self) -> None:
        """
        Normalizează eligibility trace pe turn curent și o adaugă în fereastră.
        Resetează acumulatorul pentru turnul următor.
        """
        if self._timestep_count > 0:
            mean_elig = self._elig_turn / self._timestep_count
        else:
            mean_elig = self._elig_turn.copy()

        self._elig_window.append(mean_elig)

        # Reset acumulatori turn
        self._elig_turn[:] = 0.0
        self._x_pre[:]  = 0.0
        self._x_post[:] = 0.0
        self._timestep_count = 0

    # ------------------------------------------------------------------
    # Aplicare reward → ΔW
    # ------------------------------------------------------------------

    def compute_delta_w(self, reward: float) -> np.ndarray:
        """
        Calculează ΔW = LR × reward × Σ(decay^k × elig_window[-k]).

        Returnează shape (n_post, n_pre) — aceeași formă ca W.
        """
        if not self._elig_window:
            return np.zeros((self.n_post, self.n_pre), dtype=np.float32)

        weighted_sum = np.zeros((self.n_post, self.n_pre), dtype=np.float32)
        for k, elig in enumerate(reversed(self._elig_window)):
            weighted_sum += (REWARD_DECAY ** k) * elig

        return LR * reward * weighted_sum

    # ------------------------------------------------------------------
    # Reset complet (nou episod)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._x_pre[:]   = 0.0
        self._x_post[:]  = 0.0
        self._elig_turn[:] = 0.0
        self._elig_window.clear()
        self._timestep_count = 0


# ==================================================================
# Controller R-STDP — gestionează ambele tracer-e + update greutăți
# ==================================================================

class RSTDPController:
    """
    Wrapper care leagă tracer-ele R-STDP de rețea și de engine.

    Utilizare în experiments/run.py:
        controller = RSTDPController(net)
        # per timestep (în interiorul run()):
        controller.update_timestep(spike_input, spike_hidden, spike_out)
        # la finalul turnului:
        controller.end_turn()
        # la primirea recompensei:
        controller.apply_reward(reward, net)
    """

    def __init__(self, net):
        from neuromorphic.network import N_INPUT, N_HIDDEN
        self.tracer_in  = RSTDPTracer(n_pre=N_INPUT,  n_post=N_HIDDEN)
        self.tracer_rec = RSTDPTracer(n_pre=N_HIDDEN, n_post=N_HIDDEN)
        self._net = net

    def update_timestep(self, spike_in: np.ndarray, spike_hidden: np.ndarray,
                        spike_out: np.ndarray) -> None:
        """Apelat din interiorul net.run() — sau din buclă externă."""
        self.tracer_in.update_timestep(spike_in, spike_hidden)
        self.tracer_rec.update_timestep(spike_hidden, spike_hidden)

    def end_turn(self) -> None:
        self.tracer_in.end_turn()
        self.tracer_rec.end_turn()

    def apply_reward(self, reward: float) -> None:
        dW_in  = self.tracer_in.compute_delta_w(reward)
        dW_rec = self.tracer_rec.compute_delta_w(reward)
        self._net.apply_weight_delta(dW_in, dW_rec)

        # Clipping greutăți după update
        np.clip(self._net.W_in,  W_MIN, W_MAX, out=self._net.W_in)
        np.clip(self._net.W_rec, W_MIN, W_MAX, out=self._net.W_rec)

    def reset(self) -> None:
        self.tracer_in.reset()
        self.tracer_rec.reset()
