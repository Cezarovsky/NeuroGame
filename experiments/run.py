"""
Training loop — leagă toate modulele împreună.

Rulare:
    python -m experiments.run                    # default 500 episoade
    python -m experiments.run --episodes 2000 --seed 7

Output:
    experiments/logs/run_<timestamp>.csv          — statistici per episod
    experiments/logs/weights_<timestamp>.npz      — greutăți salvate la final

CSV columns:
    episode, turns, captured, prey_reward, pred_reward,
    prey_stamina_end, prey_win_rate_100ep, pred_win_rate_100ep
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from game.arena import Arena
from game.agents import Prey, Predator, N_ACTIONS
from game.engine import GameEngine, PolicyFn
from neuromorphic.encoding import encode_state, N_INPUT
from neuromorphic.network import NeuroNet
from neuromorphic.rstdp import RSTDPController
from neuromorphic.decoder import decode_action

N_TIMESTEPS_PER_TURN = 100
LOG_DIR = Path(__file__).parent / "logs"


# ==================================================================
# Policy funcții — folosesc rețeaua + encoding + decoder
# ==================================================================

class NeuroPolicy:
    """
    Politică neuromorphică pentru un agent.
    Instanțiată separat pentru prey și predator.
    """

    def __init__(self, net: NeuroNet, rstdp: RSTDPController,
                 n_timesteps: int = N_TIMESTEPS_PER_TURN,
                 seed: int = 0):
        self.net = net
        self.rstdp = rstdp
        self.n_timesteps = n_timesteps
        self.rng = np.random.default_rng(seed)
        self._last_traces: dict | None = None

    def __call__(self, state_dict: dict) -> int:
        """Apelat de GameEngine per turn."""
        rates = encode_state(state_dict)                          # (65,)
        spikes_out, traces = self.net.run(rates, self.n_timesteps,
                                         rng=self.rng)            # (T, 32)

        # Actualizare R-STDP traces per timestep
        # (spikes_out e (T, 32) dar avem nevoie și de hidden spikes)
        spikes_hidden = traces["spikes_hidden"]   # (T, N_HIDDEN)
        spike_input_seq = self._gen_input_seq(rates)  # (T, N_INPUT)
        for t in range(self.n_timesteps):
            self.rstdp.update_timestep(
                spike_input_seq[t],
                spikes_hidden[t],
                spikes_out[t],
            )

        self.rstdp.end_turn()
        self._last_traces = traces

        return decode_action(spikes_out, rng=self.rng)

    def _gen_input_seq(self, rates: np.ndarray) -> np.ndarray:
        """
        Regenerează secvența Bernoulli de input pentru trace-uri STDP.
        Aceeași seed nu e garantată (nu contează — traces sunt oricum stochastice).
        """
        return (self.rng.random((self.n_timesteps, N_INPUT)) < rates).astype(
            np.float32
        )

    def apply_reward(self, reward: float) -> None:
        self.rstdp.apply_reward(reward)

    def reset_episode(self) -> None:
        self.net.reset()
        self.rstdp.reset()


# ==================================================================
# Utilitar: spawn aleator în arenă
# ==================================================================

def random_spawn(arena: Arena, rng: np.random.Generator,
                 min_distance: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Returnează (prey_pos, pred_pos) cu distanța minimă garantată."""
    for _ in range(1000):
        p1 = rng.uniform([2.0, 2.0], [arena.width - 2, arena.height - 2])
        p2 = rng.uniform([2.0, 2.0], [arena.width - 2, arena.height - 2])
        if np.linalg.norm(p1 - p2) >= min_distance:
            return p1, p2
    # Fallback explicit în colțuri opuse
    return (np.array([5.0, 5.0]),
            np.array([arena.width - 5, arena.height - 5]))


# ==================================================================
# Training loop
# ==================================================================

def train(n_episodes: int = 500, seed: int = DEFAULT_SEED,
          arena_w: float = 50.0, arena_h: float = 50.0) -> None:

    rng = np.random.default_rng(seed)
    arena = Arena(width=arena_w, height=arena_h)

    # Rețele independente pentru cei doi agenți
    prey_net  = NeuroNet(seed=seed)
    pred_net  = NeuroNet(seed=seed + 1)

    prey_rstdp = RSTDPController(prey_net)
    pred_rstdp = RSTDPController(pred_net)

    prey_policy = NeuroPolicy(prey_net,  prey_rstdp, seed=seed + 2)
    pred_policy = NeuroPolicy(pred_net,  pred_rstdp, seed=seed + 3)

    # CSV logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"run_{ts}.csv"
    weights_path = LOG_DIR / f"weights_{ts}.npz"

    # Sliding window win rate (100 episoade)
    prey_wins: deque[int] = deque(maxlen=100)
    pred_wins: deque[int] = deque(maxlen=100)

    print(f"Training {n_episodes} episoade | Arena {arena_w}×{arena_h} | "
          f"Log: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "turns", "captured",
            "prey_reward", "pred_reward",
            "prey_stamina_end",
            "prey_win_rate_100ep", "pred_win_rate_100ep",
        ])

        t0 = time.time()

        for ep in range(1, n_episodes + 1):
            prey_pos, pred_pos = random_spawn(arena, rng)
            prey  = Prey(prey_pos)
            pred  = Predator(pred_pos)

            prey_policy.reset_episode()
            pred_policy.reset_episode()

            engine = GameEngine(
                arena=arena,
                prey=prey,
                predator=pred,
                prey_policy=prey_policy,
                pred_policy=pred_policy,
            )

            stats = engine.run_episode()

            # Aplică reward la terminarea episodului
            # (R-STDP fereastră 3 turnuri — deja acumulate în end_turn)
            last_result = engine.history[-1]
            prey_policy.apply_reward(last_result.prey_reward)
            pred_policy.apply_reward(last_result.pred_reward)

            # Win rate tracking
            captured = int(stats.captured)
            prey_wins.append(1 - captured)   # prey câștigă dacă NU e capturat
            pred_wins.append(captured)

            prey_wr = sum(prey_wins) / len(prey_wins)
            pred_wr = sum(pred_wins) / len(pred_wins)

            writer.writerow([
                ep, stats.turns, int(stats.captured),
                round(stats.prey_total_reward, 3),
                round(stats.pred_total_reward, 3),
                round(stats.prey_stamina_end, 2),
                round(prey_wr, 3), round(pred_wr, 3),
            ])

            if ep % 50 == 0:
                elapsed = time.time() - t0
                print(f"  ep {ep:5d}/{n_episodes} | "
                      f"turns {stats.turns:4d} | "
                      f"captured {bool(stats.captured)} | "
                      f"prey_wr {prey_wr:.2f} | "
                      f"pred_wr {pred_wr:.2f} | "
                      f"{elapsed:.1f}s elapsed")

    # Salvare greutăți finale
    np.savez(
        weights_path,
        prey_W_in=prey_net.W_in,
        prey_W_rec=prey_net.W_rec,
        prey_W_out=prey_net.W_out,
        pred_W_in=pred_net.W_in,
        pred_W_rec=pred_net.W_rec,
        pred_W_out=pred_net.W_out,
    )
    print(f"\nAntrenament complet. Greutăți salvate: {weights_path}")


# ==================================================================
# Entrypoint
# ==================================================================

DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroGame training loop")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    p.add_argument("--arena-w",  type=float, default=50.0)
    p.add_argument("--arena-h",  type=float, default=50.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_episodes=args.episodes,
        seed=args.seed,
        arena_w=args.arena_w,
        arena_h=args.arena_h,
    )
