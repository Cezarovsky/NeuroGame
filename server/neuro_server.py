"""
NeuroGame WebSocket Server

Protocol:
  Python → Unity:  {"type": "init", "arena_w": 30, "arena_h": 30, "prey_pos": [...], ...}
  Python → Unity:  {"type": "step", "turn": N, "prey_pos": [...], "pred_pos": [...], ...}
  Unity  → Python: {"type": "ready"}

Flow:
  1. Unity connects → Python sends "init"
  2. Unity sends "ready" → Python runs one step → sends "step"
  3. Unity animates → sends "ready"
  4. On done=true, next "ready" triggers new episode → Python sends "init"

Rulare:
    cd NeuroGame
    python -m server.neuro_server
    python -m server.neuro_server --arena-w 20 --arena-h 20 --traps 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.arena import Arena
from game.agents import Prey, Predator
from game.engine import GameEngine
from neuromorphic.network import NeuroNet
from neuromorphic.rstdp import RSTDPController
from experiments.run import NeuroPolicy, random_spawn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("neuro_server")

PORT = 8765
DEFAULT_ARENA_W = 30.0
DEFAULT_ARENA_H = 30.0
DEFAULT_SEED = 42
DEFAULT_TRAPS = 3


# ==================================================================
# Game state manager
# ==================================================================

class NeuroGameServer:
    """
    Menține starea jocului și expune metoda step().
    O singură instanță pe server — toți clienții Unity văd același joc.
    """

    def __init__(self, arena_w: float, arena_h: float, seed: int, n_traps: int):
        self.arena = Arena(width=arena_w, height=arena_h)
        self.n_traps = n_traps
        self.rng = np.random.default_rng(seed)
        self.episode_count = 0
        self._pending_reset = False

        self._setup_policies(seed)
        self._new_episode()

    def _setup_policies(self, seed: int) -> None:
        prey_net = NeuroNet(seed=seed)
        pred_net = NeuroNet(seed=seed + 1)
        self.prey_policy = NeuroPolicy(
            prey_net, RSTDPController(prey_net), seed=seed + 2
        )
        self.pred_policy = NeuroPolicy(
            pred_net, RSTDPController(pred_net), seed=seed + 3
        )

    def _new_episode(self) -> None:
        self.arena.spawn_traps(self.n_traps, self.rng)
        prey_pos, pred_pos = random_spawn(self.arena, self.rng)
        self.prey_policy.reset_episode()
        self.pred_policy.reset_episode()
        self.engine = GameEngine(
            arena=self.arena,
            prey=Prey(prey_pos),
            predator=Predator(pred_pos),
            prey_policy=self.prey_policy,
            pred_policy=self.pred_policy,
        )
        self.episode_count += 1
        log.info(f"Episode {self.episode_count} start | "
                 f"prey={prey_pos.round(2)} pred={pred_pos.round(2)}")

    def init_msg(self) -> dict:
        return {
            "type": "init",
            "episode": self.episode_count,
            "arena_w": self.arena.width,
            "arena_h": self.arena.height,
            "prey_pos": self.engine.prey.pos.tolist(),
            "pred_pos": self.engine.predator.pos.tolist(),
            "traps": [
                {"cx": float(cx), "cy": float(cy), "r": float(r)}
                for cx, cy, r in self.arena.traps
            ],
        }

    def step(self) -> dict:
        """
        Apelat când Unity trimite 'ready'.
        Dacă episodul precedent s-a terminat, inițializează unul nou.
        """
        if self._pending_reset:
            self._pending_reset = False
            self._new_episode()
            return self.init_msg()

        result = self.engine.step()
        self.prey_policy.apply_reward(result.prey_reward)
        self.pred_policy.apply_reward(result.pred_reward)

        if result.done:
            self._pending_reset = True
            log.info(f"  → done: {result.info} | turn {result.turn}")

        return {
            "type": "step",
            "episode": self.episode_count,
            "turn": result.turn,
            "prey_pos": result.prey_pos.tolist(),
            "pred_pos": result.pred_pos.tolist(),
            "prey_stamina": float(self.engine.prey.stamina),
            "prey_reward": float(result.prey_reward),
            "pred_reward": float(result.pred_reward),
            "done": result.done,
            "info": result.info,
        }


# ==================================================================
# WebSocket handler
# ==================================================================

async def handler(websocket, game: NeuroGameServer) -> None:
    addr = websocket.remote_address
    log.info(f"Unity connected from {addr}")

    # Trimite starea inițială
    await websocket.send(json.dumps(game.init_msg()))

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"JSON invalid: {raw[:80]}")
                continue

            if msg.get("type") == "ready":
                # Rulează un turn (sau resetează episodul) și trimite înapoi
                response = game.step()
                await websocket.send(json.dumps(response))
            else:
                log.warning(f"Mesaj necunoscut: {msg}")

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning(f"Conexiune închisă cu eroare: {e}")
    finally:
        log.info(f"Unity deconectat: {addr}")


# ==================================================================
# Entrypoint
# ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroGame WebSocket server")
    p.add_argument("--port",     type=int,   default=PORT)
    p.add_argument("--arena-w",  type=float, default=DEFAULT_ARENA_W)
    p.add_argument("--arena-h",  type=float, default=DEFAULT_ARENA_H)
    p.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    p.add_argument("--traps",    type=int,   default=DEFAULT_TRAPS)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    game = NeuroGameServer(
        arena_w=args.arena_w,
        arena_h=args.arena_h,
        seed=args.seed,
        n_traps=args.traps,
    )

    async with websockets.serve(
        lambda ws: handler(ws, game),
        "localhost",
        args.port,
    ):
        log.info(f"NeuroGame server pornit pe ws://localhost:{args.port}")
        log.info(f"Arena {args.arena_w}×{args.arena_h} | traps={args.traps} | seed={args.seed}")
        await asyncio.Future()  # rulează indefinit


if __name__ == "__main__":
    asyncio.run(main())
