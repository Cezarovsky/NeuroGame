"""
NeuroGame POC — Agent Kantian cu două niveluri
Nivel 0: Reflexe hardcodate (spațiu + timp ca prioruri kantiene)
Nivel 1: Q-learning emergent (cauzalitate descoperită din experiență)

Autori: Cezar (Grădinarul) + SoraM
Data: 7 Martie 2026
"""

import pygame
import numpy as np
import argparse
import pickle
import csv
import os
import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import defaultdict


# ─── Configurare ─────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 800, 600
FPS = 60
AGENT_SIZE = 16
OBJ_SIZE = 12

# Culori
BLACK      = (10, 10, 20)
BLUE       = (60, 130, 255)
RED        = (220, 60, 60)
GRAY       = (140, 140, 140)
YELLOW     = (255, 220, 0)
WHITE      = (240, 240, 240)
GREEN      = (60, 200, 80)
ORANGE     = (255, 140, 0)

# ─── Nivel 0: Constante kantiene ─────────────────────────────────────────────

LOOMING_DANGER_THRESHOLD = 0.08   # looming_index > asta → reflex evadare
LOOMING_WARN_THRESHOLD   = 0.04   # warning, Nivel 1 e alertat
REFLEX_SPEED             = 6.0    # viteza reflexului kantian (px/frame)
SENSOR_RADIUS            = 300    # câmpul vizual al agentului (px)


# ─── Obiecte din mediu ───────────────────────────────────────────────────────

@dataclass
class WorldObject:
    obj_id: int
    x: float
    y: float
    vx: float
    vy: float
    dangerous: bool         # dacă lovirea = penalizare
    pursuer: bool = False   # urmărește activ agentul (theory of mind test)
    ax: float = 0.0         # accelerare (înnăscută sau emergentă)
    ay: float = 0.0
    max_speed: float = 4.0
    radius: int = OBJ_SIZE

    def update(self, agent_x: float, agent_y: float):
        if self.pursuer:
            # Prădătorul se reorientează spre agent
            dx = agent_x - self.x
            dy = agent_y - self.y
            dist = max(1.0, (dx**2 + dy**2)**0.5)
            self.ax = (dx / dist) * 0.15
            self.ay = (dy / dist) * 0.15

        self.vx = np.clip(self.vx + self.ax, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy + self.ay, -self.max_speed, self.max_speed)
        self.x += self.vx
        self.y += self.vy

        # Bounce la margini
        if self.x < self.radius or self.x > WIDTH - self.radius:
            self.vx *= -1
            self.ax *= -1
        if self.y < self.radius or self.y > HEIGHT - self.radius:
            self.vy *= -1
            self.ay *= -1

    @property
    def pos(self) -> Tuple[float, float]:
        return self.x, self.y

    def distance_to(self, ax: float, ay: float) -> float:
        return ((self.x - ax)**2 + (self.y - ay)**2)**0.5


# ─── Nivel 0: Sensor kantian ─────────────────────────────────────────────────

class KantianSensor:
    """
    Calculează looming index pentru fiecare obiect vizibil.
    Looming = viteza de apropiere / distanță curentă
    Reprezintă percepția de timp-spațiu ca prior înnăscut.
    """

    def __init__(self, threshold_danger: float = LOOMING_DANGER_THRESHOLD,
                 threshold_warn: float = LOOMING_WARN_THRESHOLD):
        self.threshold_danger = threshold_danger
        self.threshold_warn = threshold_warn
        self.reflex_active = False
        self.reflex_direction = np.array([0.0, 0.0])
        self.reflex_frames_left = 0

    def compute_looming(self, agent_x: float, agent_y: float,
                        obj: WorldObject) -> float:
        """
        Looming index = rata de schimbare relativă a distanței.
        Pozitiv = obiectul se apropie.
        """
        dist = obj.distance_to(agent_x, agent_y)
        if dist < 1.0:
            return float('inf')

        # Proiecția vitezei relative pe axa agent-obiect
        dx = agent_x - obj.x
        dy = agent_y - obj.y
        norm = max(1.0, dist)
        # Viteza de apropiere (pozitiv = se apropie)
        approach_speed = (obj.vx * (-dx/norm) + obj.vy * (-dy/norm))
        return approach_speed / dist

    def process(self, agent_x: float, agent_y: float,
                objects: List[WorldObject]) -> Tuple[bool, np.ndarray, List[float]]:
        """
        Returnează:
          - reflex_triggered: bool
          - reflex_direction: vector evadare
          - looming_values: [looming per obiect]
        """
        looming_values = []
        max_looming = 0.0
        danger_obj = None

        for obj in objects:
            dist = obj.distance_to(agent_x, agent_y)
            if dist > SENSOR_RADIUS:
                looming_values.append(0.0)
                continue

            loom = self.compute_looming(agent_x, agent_y, obj)
            looming_values.append(loom)

            if loom > max_looming:
                max_looming = loom
                danger_obj = obj

        # Reflex kantian: dacă orice obiect depășește threshold
        if max_looming > self.threshold_danger and danger_obj is not None:
            self.reflex_active = True
            self.reflex_frames_left = 8  # durata reflexului

            # Direcția de evadare = departe de obiectul periculos
            dx = agent_x - danger_obj.x
            dy = agent_y - danger_obj.y
            norm = max(1.0, (dx**2 + dy**2)**0.5)
            self.reflex_direction = np.array([dx / norm, dy / norm])
        else:
            if self.reflex_frames_left > 0:
                self.reflex_frames_left -= 1
            else:
                self.reflex_active = False
                self.reflex_direction = np.array([0.0, 0.0])

        return self.reflex_active, self.reflex_direction.copy(), looming_values


# ─── Nivel 1: Q-Learning tabular ─────────────────────────────────────────────

class QLearningAgent:
    """
    Agent care descoperă cauzalitatea din experiență.
    Nu i se spune ce e pericol — descoperă din penalizări.
    """

    ACTIONS = [
        np.array([0.0,  0.0]),   # stai
        np.array([0.0, -1.0]),   # sus
        np.array([0.0,  1.0]),   # jos
        np.array([-1.0, 0.0]),   # stânga
        np.array([1.0,  0.0]),   # dreapta
    ]
    ACTION_NAMES = ["stai", "sus", "jos", "stânga", "dreapta"]

    def __init__(self, lr: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.last_state = None
        self.last_action = 0

    def discretize_state(self, agent_x: float, agent_y: float,
                         looming_values: List[float],
                         level0_active: bool) -> tuple:
        """
        Discretizăm starea pentru Q-table.
        Starea include: poziție grosieră + looming maxim + reflex activ.
        """
        # Zonă spațială (grila 8x6)
        gx = int(np.clip(agent_x / (WIDTH / 8), 0, 7))
        gy = int(np.clip(agent_y / (HEIGHT / 6), 0, 5))

        # Looming maxim (5 nivele)
        max_loom = max(looming_values) if looming_values else 0.0
        loom_bucket = min(4, int(max_loom / (LOOMING_DANGER_THRESHOLD / 2)))

        return (gx, gy, loom_bucket, int(level0_active))

    def choose_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(self.ACTIONS) - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool):
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.lr * (target - current_q)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'q_table': dict(self.q_table), 'epsilon': self.epsilon}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.ACTIONS)), data['q_table'])
            self.epsilon = data['epsilon']


# ─── Mediu de joc ────────────────────────────────────────────────────────────

class NeuroGameEnv:

    def __init__(self, env_variant: str = 'default', seed: int = 42):
        self.variant = env_variant
        self.rng = np.random.RandomState(seed)
        self.objects: List[WorldObject] = []
        self.agent_x = WIDTH / 2
        self.agent_y = HEIGHT / 2
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.alive = True
        self.frame = 0
        self.obj_counter = 0

        # Speed multiplier per variant
        self.speed_mult = {'default': 1.0, 'fast': 3.0, 'slow': 0.3,
                           'gravity': 1.0}.get(env_variant, 1.0)

    def _spawn_objects(self, n_neutral: int = 4, n_dangerous: int = 2,
                       n_pursuers: int = 1):
        self.objects = []
        for _ in range(n_neutral):
            self._add_object(dangerous=False, pursuer=False)
        for _ in range(n_dangerous):
            self._add_object(dangerous=True, pursuer=False)
        for _ in range(n_pursuers):
            self._add_object(dangerous=True, pursuer=True)

    def _add_object(self, dangerous: bool, pursuer: bool):
        # Spawn pe margini (nu în centru)
        side = self.rng.randint(4)
        if side == 0:   x, y = self.rng.uniform(0, WIDTH), 10.0
        elif side == 1: x, y = self.rng.uniform(0, WIDTH), HEIGHT - 10.0
        elif side == 2: x, y = 10.0, self.rng.uniform(0, HEIGHT)
        else:           x, y = WIDTH - 10.0, self.rng.uniform(0, HEIGHT)

        base_speed = self.rng.uniform(0.5, 2.5) * self.speed_mult
        angle = self.rng.uniform(0, 2 * np.pi)
        vx = np.cos(angle) * base_speed
        vy = np.sin(angle) * base_speed

        # Obiectele periculoase accelerează spre agent
        ax = ay = 0.0
        if dangerous and not pursuer:
            ax = self.rng.uniform(0.02, 0.08)
            ay = self.rng.uniform(0.02, 0.08)
            if self.rng.random() > 0.5: ax *= -1
            if self.rng.random() > 0.5: ay *= -1

        self.obj_counter += 1
        self.objects.append(WorldObject(
            obj_id=self.obj_counter,
            x=x, y=y, vx=vx, vy=vy,
            dangerous=dangerous, pursuer=pursuer,
            ax=ax, ay=ay,
            max_speed=4.0 * self.speed_mult
        ))

    def reset(self) -> dict:
        self.agent_x = WIDTH / 2
        self.agent_y = HEIGHT / 2
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.alive = True
        self.frame = 0
        self._spawn_objects()
        return self._get_state()

    def _get_state(self) -> dict:
        return {
            'agent_x': self.agent_x,
            'agent_y': self.agent_y,
            'objects': self.objects,
            'frame': self.frame,
        }

    def step(self, move_vector: np.ndarray) -> Tuple[dict, float, bool]:
        """
        Aplică mișcarea agentului (din Nivel 0 sau Nivel 1).
        Returnează: (state, reward, done)
        """
        if not self.alive:
            return self._get_state(), 0.0, True

        # Mișcare agent
        speed = REFLEX_SPEED
        self.agent_x = np.clip(self.agent_x + move_vector[0] * speed,
                               AGENT_SIZE, WIDTH - AGENT_SIZE)
        self.agent_y = np.clip(self.agent_y + move_vector[1] * speed,
                               AGENT_SIZE, HEIGHT - AGENT_SIZE)

        # Gravity variant
        if self.variant == 'gravity':
            self.agent_vy += 0.1
            self.agent_y = np.clip(self.agent_y + self.agent_vy,
                                   AGENT_SIZE, HEIGHT - AGENT_SIZE)
            if self.agent_y >= HEIGHT - AGENT_SIZE:
                self.agent_vy = 0.0

        # Update obiecte
        for obj in self.objects:
            obj.update(self.agent_x, self.agent_y)

        # Coliziune
        reward = 0.1  # reward pentru supraviețuire
        done = False
        for obj in self.objects:
            dist = obj.distance_to(self.agent_x, self.agent_y)
            if dist < (AGENT_SIZE + obj.radius) * 0.8:
                if obj.dangerous:
                    reward = -100.0
                    done = True
                    self.alive = False
                    break

        self.frame += 1
        if self.frame > 3600:  # max 60 secunde la 60fps
            done = True

        return self._get_state(), reward, done


# ─── Game loop principal ──────────────────────────────────────────────────────

class NeuroGame:

    def __init__(self, args):
        self.args = args
        self.visual = (args.mode == 'visual')
        self.level0_only = args.level0_only
        self.level1_only = args.level1_only

        self.env = NeuroGameEnv(
            env_variant=getattr(args, 'env_variant', 'default'),
            seed=args.seed
        )
        self.sensor = KantianSensor()
        self.agent_l1 = QLearningAgent()

        os.makedirs('logs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)

        self.stats = []

        if self.visual:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("NeuroGame — Kant + Piaget")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('monospace', 14)
            self.font_big = pygame.font.SysFont('monospace', 18, bold=True)

    def _compute_move(self, state: dict,
                      looming_values: List[float],
                      reflex_active: bool,
                      reflex_dir: np.ndarray) -> Tuple[np.ndarray, int, tuple]:
        """
        Decide mișcarea combinând Nivel 0 și Nivel 1.
        Nivel 0 are prioritate absolută dacă reflexul e activ.
        """
        l1_state = self.agent_l1.discretize_state(
            state['agent_x'], state['agent_y'],
            looming_values, reflex_active
        )

        if self.level0_only:
            # Doar reflexe kantiene
            if reflex_active:
                return reflex_dir, -1, l1_state
            return np.array([0.0, 0.0]), -1, l1_state

        if self.level1_only:
            # Doar Q-learning, fără Nivel 0
            action_idx = self.agent_l1.choose_action(l1_state)
            return self.agent_l1.ACTIONS[action_idx], action_idx, l1_state

        # NeuroGame: Nivel 0 are prioritate
        if reflex_active:
            return reflex_dir, -1, l1_state

        action_idx = self.agent_l1.choose_action(l1_state)
        return self.agent_l1.ACTIONS[action_idx], action_idx, l1_state

    def _draw(self, state: dict, looming_values: List[float],
              reflex_active: bool, episode: int, ep_reward: float):
        self.screen.fill(BLACK)

        # Obiecte
        for i, obj in enumerate(state['objects']):
            color = RED if obj.dangerous else GRAY
            if obj.pursuer:
                color = ORANGE
            pygame.draw.circle(self.screen, color,
                               (int(obj.x), int(obj.y)), obj.radius)

            # Looming overlay
            if i < len(looming_values) and looming_values[i] > LOOMING_WARN_THRESHOLD:
                intensity = min(255, int(looming_values[i] / LOOMING_DANGER_THRESHOLD * 200))
                warn_color = (intensity, intensity, 0)
                pygame.draw.circle(self.screen, warn_color,
                                   (int(obj.x), int(obj.y)), obj.radius + 4, 2)

        # Agent
        agent_color = YELLOW if reflex_active else BLUE
        pygame.draw.circle(self.screen, agent_color,
                           (int(state['agent_x']), int(state['agent_y'])),
                           AGENT_SIZE)

        # HUD
        mode_str = "L0-only" if self.level0_only else ("L1-only" if self.level1_only else "NeuroGame")
        lines = [
            f"Episode: {episode}  |  Mode: {mode_str}",
            f"Reward: {ep_reward:.1f}  |  Frame: {state['frame']}",
            f"Nivel 0 activ: {'DA ⚡' if reflex_active else 'nu'}",
            f"ε (esploration): {self.agent_l1.epsilon:.3f}",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, WHITE)
            self.screen.blit(surf, (10, 10 + i * 18))

        # Legendă
        legend = [("■ Neutral", GRAY), ("■ Periculos", RED),
                  ("■ Prădător", ORANGE), ("■ Agent", BLUE),
                  ("■ Reflex activ", YELLOW)]
        for i, (txt, col) in enumerate(legend):
            surf = self.font.render(txt, True, col)
            self.screen.blit(surf, (WIDTH - 130, 10 + i * 18))

        pygame.display.flip()

    def run(self):
        print(f"[NeuroGame] Start — {self.args.episodes} episoade, mode={self.args.mode}")
        if not self.level1_only:
            print(f"[NeuroGame] Nivel 0 kantian ACTIV (threshold={LOOMING_DANGER_THRESHOLD})")

        log_path = f"logs/{self.args.log_prefix}_stats.csv"
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'frames', 'reward', 'reflex_triggers', 'epsilon'])

        for episode in range(1, self.args.episodes + 1):
            state = self.env.reset()
            ep_reward = 0.0
            reflex_triggers = 0
            done = False
            prev_l1_state = None
            prev_action = None

            while not done:
                # Nivel 0: calcul senzori kantieni
                reflex_active, reflex_dir, looming_values = self.sensor.process(
                    state['agent_x'], state['agent_y'], state['objects']
                )
                if reflex_active:
                    reflex_triggers += 1

                # Decide mișcare
                move_vec, action_idx, l1_state = self._compute_move(
                    state, looming_values, reflex_active, reflex_dir
                )

                # Update Q-table cu experiența anterioară
                if prev_l1_state is not None and prev_action is not None and not self.level0_only:
                    self.agent_l1.update(
                        prev_l1_state, prev_action, 0.1,
                        l1_state, False
                    )

                prev_l1_state = l1_state
                prev_action = action_idx if action_idx >= 0 else 0

                # Step mediu
                next_state, reward, done = self.env.step(move_vec)
                ep_reward += reward

                # Update final Q (cu reward real)
                if (not self.level0_only) and prev_action is not None:
                    next_reflex, _, next_looming = self.sensor.process(
                        next_state['agent_x'], next_state['agent_y'],
                        next_state['objects']
                    )
                    next_l1_state = self.agent_l1.discretize_state(
                        next_state['agent_x'], next_state['agent_y'],
                        next_looming, next_reflex
                    )
                    self.agent_l1.update(l1_state, prev_action, reward,
                                         next_l1_state, done)

                state = next_state

                # Visual
                if self.visual:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                    self._draw(state, looming_values, reflex_active, episode, ep_reward)
                    self.clock.tick(FPS)

            # Log episod
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, state['frame'], ep_reward,
                                  reflex_triggers, self.agent_l1.epsilon])

            if episode % 100 == 0:
                print(f"[Ep {episode:5d}] frames={state['frame']:4d}  "
                      f"reward={ep_reward:8.1f}  reflex={reflex_triggers}  "
                      f"ε={self.agent_l1.epsilon:.4f}")

            # Checkpoint
            if episode % self.args.save_every == 0 and not self.level0_only:
                ckpt_path = f"checkpoints/{self.args.log_prefix}_ep{episode}.pkl"
                self.agent_l1.save(ckpt_path)
                print(f"[NeuroGame] Checkpoint salvat: {ckpt_path}")

        print(f"[NeuroGame] Terminat. Log: {log_path}")
        if self.visual:
            pygame.quit()


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NeuroGame — Agent Kantian")
    p.add_argument('--mode', choices=['visual', 'headless'], default='visual')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--level0-only', action='store_true',
                   help='Doar reflexe kantiene, fără Q-learning')
    p.add_argument('--level1-only', action='store_true',
                   help='Doar Q-learning, fără Nivel 0')
    p.add_argument('--env-variant', choices=['default', 'fast', 'slow', 'gravity'],
                   default='default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--log-prefix', type=str, default='neurogame')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    game = NeuroGame(args)
    game.run()
