"""
NeuroGame Visualizer — Pygame integrat cu SNN/R-STDP engine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CE FACE ACEST FIȘIER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este fișierul central al simulării — combină totul:
  • Pygame: fereastra vizuală, input tastatură, render
  • SNN (NeuroNet): rețelele neuronale spiking ale agenților
  • R-STDP (RSTDPController): algoritmul de învățare
  • GameEngine: fizica simulării (mișcare, coliziuni, capturi)
  • GrassManager: resursele de iarbă și reward-ul de hrănire
  • Level 0 behaviors: comportamente hardcodate (patrol, foraging)
  • Plateau detector: monitorizare consolidare SNN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARHITECTURA DECIZIEI (per turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  engine.step()
      │
      ├─► prey_policy_l0(state_dict)      ← apelat de engine
      │       │
      │       ├─ [pericol?] ──► prey_policy(state_dict)   ← SNN decide
      │       │                     │
      │       │                encode_state() → 73 rate-uri
      │       │                net.run() → 100 timesteps LIF
      │       │                decode_action() → 1 din 32 acțiuni
      │       │
      │       └─ [safe?] ──► L0: linger / sensing iarbă / waypoint
      │
      └─► pred_policy_eps(state_dict)     ← apelat de engine
              │
              ├─ [epsilon?] ──► acțiune random (0-31)
              └─ pred_policy_l0() ──► [aproape?] SNN / [departe?] patrol

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZONE RADIANT (distanță Fenrir–Umbra)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE       < 1.5u  → captură (episod terminat)
  RADIATIVE  < 5.0u  → pericol iminent
  CONVECTION < 10.0u → pericol în creștere
  CORONA     ≥ 10.0u → safe

ZONE SENSING IARBĂ (distanță Umbra–patch):
  CORE       < 1.5u  → mănâncă + linger
  RADIATIVE  < 4.0u  → vede exact (noise=1)
  CONVECTION < 8.0u  → ~50% acuratețe
  CORONA     < 15.0u → miros vag (noise=12)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cd NeuroGame
  ../.venv/bin/python visualizer.py
  ../.venv/bin/python visualizer.py --episodes 2000 --speed 10
  ../.venv/bin/python visualizer.py --headless   # fără fereastră
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pygame

# ─── Path setup ───────────────────────────────────────────────────
# visualizer.py e în NeuroGame/ și importă din subdirectoarele:
#   game/        → Arena, Prey, Predator, GameEngine
#   neuromorphic/→ NeuroNet, RSTDPController
#   experiments/ → NeuroPolicy, random_spawn
#
# sys.path.insert(0, ...) adaugă directorul curent PRIMUL în lista de căutare,
# astfel încât "from game.arena import Arena" găsește game/arena.py local,
# nu un eventual pachet "game" instalat global în sistem.
sys.path.insert(0, str(Path(__file__).parent))

from game.arena import Arena
from game.agents import Prey, Predator
from game.engine import GameEngine
from neuromorphic.network import NeuroNet
from neuromorphic.rstdp import RSTDPController
from experiments.run import NeuroPolicy, random_spawn


# ==================================================================
# CONSTANTE VIZUALE
# ==================================================================
#
# Toate constantele vizuale sunt în pixeli sau tuple RGB/RGBA.
# Schimbarea lor afectează DOAR afișarea, nu simularea.
# ==================================================================

WIN_W, WIN_H = 700, 700
# Dimensiunea ferestrei Pygame. Ambele dimensiuni egale → arenă pătrată.
# Dacă schimbi WIN_W ≠ WIN_H, CoordMapper va distorsiona axa X vs Y.

ARENA_MARGIN = 40
# Spațiu liber (pixeli) între marginea ferestrei și arenă.
# Necesar pentru HUD (statistici sus-stânga) și legendă (jos-stânga).

FPS_CAP = 60
# Frame rate maxim. clock.tick(FPS_CAP) limitează render-ul.
# Nu afectează viteza simulării — speed (turnuri/frame) controlează asta.

# ─── Culori RGB (Red, Green, Blue) și RGBA (+ Alpha=transparență) ─
C_BG         = (15,  20,  30)       # negru-albăstrui — fundal fereastră
C_ARENA      = (25,  35,  50)       # albastru închis — fundalul arenei
C_WALL       = (60,  80, 110)       # albastru mediu — bordura arenei
C_PREY       = (80, 200,  80)       # verde — Umbra (pradă/iepure)
C_PRED       = (220,  60,  60)      # roșu — Fenrir (prădător/lup)
C_TRAP       = (180,  40,  40)      # roșu închis — capcane
C_TRAP_ALPHA = 80                   # 0=invizibil, 255=opac; 80≈31% opac
C_ZONE_CONV  = (255, 200,  50,  25) # galben foarte transparent (zona Convection)
C_ZONE_RAD   = (255, 100,  30,  50) # portocaliu semi-transparent (zona Radiative)
C_TEXT       = (200, 210, 230)      # alb-albăstrui — text principal HUD
C_TEXT_DIM   = (100, 120, 150)      # gri-albăstrui — text secundar HUD
C_GRASS      = (50,  180,  50)      # verde mediu — patch iarbă activ
C_GRASS_EATEN= (30,  100,  30)      # verde închis — patch mâncat (în regen)
C_CAPTURE    = (255, 255, 100)      # galben — mesaj "CAPTURAT!"
C_TIMEOUT    = (100, 150, 255)      # albastru — mesaj "TIMEOUT"


# ==================================================================
# CONSTANTE DE SIMULARE
# ==================================================================
#
# Unitățile de distanță sunt "unități Python" — sistemul de coordonate
# intern al simulării, unde arena e 30.0 × 30.0 unități.
# Pixelii se calculează dinamic în CoordMapper.to_px().
# ==================================================================

# ─── Zone radiant (distanță Fenrir–Umbra) ─────────────────────────
# Inspirate din structura solară. Sunt distanțe ABSOLUTE, nu relative la arenă.
# Motivație biologică: un iepure percepe pericolul la aceeași distanță
# indiferent dacă câmpul e mic sau mare — senzorul e fix, nu contextual.
CORE_R       = 1.5   # sub această distanță → Fenrir capturează Umbra
RADIATIVE_R  = 5.0   # sub această distanță → L0 activează urmărirea activă
CONVECTION_R = 10.0  # sub această distanță → L0 cedează controlul SNN-ului pentru Umbra

# ─── Parametri iarbă ──────────────────────────────────────────────
N_GRASS      = 4     # numărul de patch-uri simultane în arenă
               # Mai multe patch-uri = mai mult reward disponibil = Umbra mai motivată
               # Prea multe = Fenrir nu mai are avantaj (Umbra mereu saturată)

GRASS_CORE_R = 1.5   # raza de mâncat (unități Python)
               # Umbra trebuie să intre FIZIC în acest cerc pentru a mânca
               # Același ca CORE_R — coincidență intenționată (simétrie biologică)

GRASS_RAD_R  = 4.0   # raza de vizibilitate clară a ierbii
               # Sub această distanță: Umbra "vede" exact unde e iarba (noise=1)

GRASS_CONV_R = 8.0   # raza de detectare incertă a ierbii
               # Sub această distanță: ~50% șansă de direcție corectă

GRASS_REGEN    = 40    # turnuri de regenerare după ce un patch e mâncat
               # 40 turnuri ≈ jumătate dintr-un episod tipic (max_turns=500 în engine)

GRASS_EAT_TURNS = 3   # turnuri consecutive în CORE pentru a consuma un patch
               # Umbra trebuie să RĂMÂNĂ în raza de 1.5u timp de 3 turnuri consecutive.
               # Dacă iese înainte → eating_progress se resetează (mâncat întrerupt).
               # Reglează raritatea ierbii → presiune de foraging

# ─── Capcane ──────────────────────────────────────────────────────
UMBRA_R    = 0.5
# Raza "corpului" Umbrei în unități Python.
# Folosit ca unitate de referință pentru dimensiunea capcanelor.

TRAP_SIZES = [UMBRA_R, UMBRA_R * 1.5, UMBRA_R * 1.5]
# Lista razelor celor 3 capcane:
#   UMBRA_R        = 0.5  → capcană mică (exact cât Umbra — greu de evitat)
#   UMBRA_R * 1.5  = 0.75 → capcană medie (×2 exemplare)
# Dacă Umbra intră într-o capcană → episod terminat, prey reward = -10

# ─── Epsilon-greedy Fenrir ────────────────────────────────────────
# Epsilon = probabilitatea de a alege o acțiune RANDOM în loc de politica L0/SNN.
# Scade exponențial per episod: ε_nou = max(MIN, ε_vechi × DECAY)
#
# La ep 0:    ε = 0.40  → 40% random, 60% politică
# La ep 500:  ε ≈ 0.40 × 0.999^500 ≈ 0.242
# La ep 1000: ε ≈ 0.40 × 0.999^1000 ≈ 0.147
# La ep 2300: ε ≈ 0.05  (atinge minimul)
PRED_EPSILON_START = 0.4
PRED_EPSILON_MIN   = 0.05
PRED_EPSILON_DECAY = 0.999

# ─── Parametri arenă și training ──────────────────────────────────
ARENA_W = 30.0   # lățimea arenei în unități Python
ARENA_H = 30.0   # înălțimea arenei
N_TRAPS = 3      # numărul de capcane (folosit în argparse; intern suprascriem cu TRAP_SIZES)
SEED    = 42     # seed pentru reproductibilitate completă a simulării


# ==================================================================
# CLASA GrassPatch — un singur patch de iarbă
# ==================================================================

class GrassPatch:
    """
    Reprezintă un singur patch de iarbă în arenă.

    Atribute:
        pos (np.ndarray shape (2,)):
            Coordonatele [x, y] ale patch-ului în spațiul Python.
            dtype=float pentru compatibilitate cu np.linalg.norm().

        alive (bool):
            True  → patch vizibil, poate fi mâncat de Umbra
            False → patch mâncat, invizibil, se regenerează

        regen_timer (int):
            Numărul de turnuri rămase până când patch-ul revine la alive=True.
            Pornește de la GRASS_REGEN (40) când alive devine False.
            Decrementat cu 1 per turn în GrassManager.update().
    """
    def __init__(self, x: float, y: float):
        self.pos            = np.array([x, y], dtype=float)
        self.alive          = True
        self.regen_timer    = 0
        self.eating_progress = 0  # turnuri consecutive în care Umbra e în CORE
                                  # 0 = nu se mănâncă; GRASS_EAT_TURNS = consumat


# ==================================================================
# CLASA GrassManager — gestionează toate patch-urile
# ==================================================================

class GrassManager:
    """
    Gestionează ciclul de viață al tuturor patch-urilor de iarbă.

    De ce există separat față de Arena?
        Arena gestionează structura fixă (pereți, capcane).
        GrassManager gestionează resurse dinamice (iarbă) cu logică proprie
        de spawn, mâncat și regenerare.

    Fluxul per episod:
        __init__ sau reset() → _spawn_all() → N_GRASS patch-uri noi
        Per turn: update(prey_pos) → verifică mâncat + regenerare + reward
        active_positions() → lista pozițiilor pentru SNN encoding
    """

    def __init__(self, n: int, arena: Arena, rng: np.random.Generator):
        self.n     = n       # numărul de patch-uri simultane
        self.arena = arena   # referință la arenă (pentru check_trap + dimensiuni)
        self.rng   = rng     # generator de numere aleatoare (partajat cu main loop)
        self.patches: list[GrassPatch] = []
        self._spawn_all()

    def _spawn_one(self) -> GrassPatch:
        """
        Generează un singur patch într-o poziție validă.

        Algoritmul:
            Încearcă de maximum 500 ori să găsească o poziție care:
            - Nu e pe o capcană (arena.check_trap returnează False)
            - E la cel puțin 2 unități de orice perete

            Dacă după 500 încercări nu găsim → fallback în centrul arenei.
            (Cazul apare doar dacă capcanele acoperă aproape toată arena — practic imposibil)

        De ce margine 2.0?
            Patch-urile la < 2u de perete ar fi greu accesibile pentru Umbra
            fără să se atingă de pereți — experiență frustrantă.
        """
        for _ in range(500):
            x = self.rng.uniform(2.0, self.arena.width  - 2.0)
            y = self.rng.uniform(2.0, self.arena.height - 2.0)
            pos = np.array([x, y])
            if not self.arena.check_trap(pos):
                return GrassPatch(x, y)
        # Fallback: centrul arenei (în cazuri extreme)
        return GrassPatch(self.arena.width / 2, self.arena.height / 2)

    def _spawn_all(self):
        """Creează N_GRASS patch-uri noi, înlocuind orice patch existent."""
        self.patches = [self._spawn_one() for _ in range(self.n)]

    def reset(self):
        """
        Apelat la ÎNCEPUTUL fiecărui episod.
        Respawn-ează toate patch-urile în poziții noi aleatorii.
        Poziția ierbii se schimbă între episoade (spre deosebire de capcane care sunt fixe).
        """
        self._spawn_all()

    def update(self, prey_pos: np.ndarray) -> tuple[float, "np.ndarray | None"]:
        """
        Apelat O DATĂ PER TURN, DUPĂ engine.step().

        Ordinea apelurilor în main loop:
            result = engine.step()          ← agenții se mișcă
            grass_reward, eaten_pos = grass_mgr.update(engine.prey.pos)

        De ce DUPĂ engine.step()?
            engine.step() apelează politicile și mișcă agenții.
            Verificăm iarbă cu NOUA poziție a Umbrei (post-mișcare).

        ── Mecanica de mâncat (progresivă, GRASS_EAT_TURNS = 3 turnuri) ──

        Umbra NU consumă instant un patch când intră în CORE.
        Trebuie să RĂMÂNĂ în CORE (dist < 1.5u) timp de GRASS_EAT_TURNS turnuri consecutive.

            eating_progress = 0     → Umbra nu era în zonă sau a plecat
            eating_progress = 1,2   → în curs de mâncat (patch încă alive → vizibil)
            eating_progress >= 3    → consumat: alive=False, reward=+5.0

        Dacă Umbra IESE din CORE înainte de GRASS_EAT_TURNS:
            eating_progress se resetează la 0 → mâncatul e întrerupt.

        ── Reward-uri de proximitate (rămân) ──

            Dacă dist < GRASS_RAD_R (4.0u):  reward += 0.3
            Dacă dist < GRASS_CONV_R (8.0u): reward += 0.05

        ── Returnează ──

            (reward: float, eaten_pos: np.ndarray | None)
            eaten_pos = poziția patch-ului consumat în acest turn (pentru linger)
            eaten_pos = None dacă niciun patch nu a fost consumat
        """
        reward    = 0.0
        eaten_pos = None

        for p in self.patches:
            if not p.alive:
                p.regen_timer -= 1
                if p.regen_timer <= 0:
                    new   = self._spawn_one()
                    p.pos = new.pos
                    p.alive          = True
                    p.eating_progress = 0
                continue  # ← patch mort: nu verificăm distanța

            dist = float(np.linalg.norm(prey_pos - p.pos))

            if dist < GRASS_CORE_R:
                # Umbra e în zonă: avansăm eating_progress
                p.eating_progress += 1
                if p.eating_progress >= GRASS_EAT_TURNS:
                    # Consumat! Patch-ul dispare.
                    eaten_pos         = p.pos.copy()
                    p.alive           = False
                    p.regen_timer     = GRASS_REGEN
                    p.eating_progress = 0
                    reward           += 5.0
                # Cât timp mănâncă: nu dăm și reward de proximitate (se suprapun)
            else:
                # Umbra a plecat din CORE → mâncat întrerupt
                p.eating_progress = 0
                if dist < GRASS_RAD_R:
                    reward += 0.3
                elif dist < GRASS_CONV_R:
                    reward += 0.05

        return reward, eaten_pos

    def active_positions(self) -> list[tuple[float, float]]:
        """
        Returnează lista pozițiilor patch-urilor VIVE ca tuple-uri (x, y).

        Folosit în două locuri:
            1. prey_policy_l0: pentru sensing gradient (distanță față de cel mai apropiat)
            2. SNN encoding (encoding.py → _grass_ring): pentru cei 8 neuroni direcționali

        Patch-urile moarte NU sunt incluse — Umbra nu "vede" iarbă care nu există.
        """
        return [(p.pos[0], p.pos[1]) for p in self.patches if p.alive]


# ==================================================================
# FUNCȚIA compute_rewards — modelul de reward radiant
# ==================================================================

def compute_rewards(prey_pos, pred_pos, captured: bool, in_trap: bool):
    """
    Calculează reward-ul per turn pentru AMBII agenți.

    PARAMETRI:
        prey_pos (np.ndarray): poziția Umbrei [x, y]
        pred_pos (np.ndarray): poziția lui Fenrir [x, y]
        captured (bool): True dacă episodul tocmai s-a terminat prin captură
        in_trap  (bool): True dacă episodul tocmai s-a terminat prin capcană

    RETURNEAZĂ: (prey_reward, pred_reward) — tuple de două float-uri

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LOGICA REWARD-ULUI
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Capcană (in_trap=True):
        prey: -10.0  → penalizare maximă (Umbra a murit prost)
        pred:   0.0  → Fenrir e neutru (nu el a prins-o)

    Captură (captured=True):
        prey: -10.0  → penalizare maximă
        pred: +10.0  → reward maxim (Fenrir a câștigat)

    Distanță < RADIATIVE_R (5u) — pericol iminent:
        prey:  -2.0  → penalizare mare pe fiecare turn (urgență de fugă)
        pred:  +1.0  → reward per turn (Fenrir e pe urmele Umbrei)

    Distanță < CONVECTION_R (10u) — pericol în creștere:
        prey:  -0.5  → penalizare medie
        pred:  +0.3  → reward mediu

    Distanță ≥ CONVECTION_R (10u) — zona Corona:
        prey:  +0.1  → reward mic (Umbra e în siguranță → relaxare)
        pred:  -0.1  → PENALIZARE pentru Fenrir!

    De ce penalizăm Fenrir în Corona?
        Fără penalizare: Fenrir descoperă că cel mai safe e să nu facă nimic
        (reward=0 stând pe loc vs risc de -10 dacă Umbra scapă).
        Cu penalizare -0.1/turn: după 100 turnuri → -10, echivalent cu o ratare.
        Forțează URGENȚA de vânătoare în R-STDP.

    De ce reward dens (per turn) și nu sparse (doar la captură)?
        Reward sparse (0 mereu, +100 la captură): R-STDP nu are semnal
        suficient pentru a converge — gradient prea rar.
        Reward dens (fiecare turn): semnal constant → convergență mai rapidă.
        Analogie: să înveți să meargă cu "bravo" la fiecare pas
        vs "bravo" doar când ajungi la destinație.
    """
    if in_trap:
        return -10.0, 0.0
    if captured:
        return -10.0, +10.0

    # np.linalg.norm pe diferența vectorilor = distanța Euclidiană în 2D
    # sqrt((prey_x - pred_x)² + (prey_y - pred_y)²)
    dist = float(np.linalg.norm(prey_pos - pred_pos))

    if dist < RADIATIVE_R:
        return -2.0, +1.0
    elif dist < CONVECTION_R:
        return -0.5, +0.3
    else:
        return +0.1, -0.1


# ==================================================================
# CLASA CoordMapper — conversie sisteme de coordonate
# ==================================================================

class CoordMapper:
    """
    Convertește coordonate din spațiul simulării (Python) în pixeli Pygame.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DOUĂ SISTEME DE COORDONATE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Spațiul Python (simulare):
        Origine: colțul STÂNGA-JOS
        X crește spre dreapta: 0 → ARENA_W (30.0)
        Y crește în sus:       0 → ARENA_H (30.0)
        Exemplu: (0,0) = colț stânga-jos, (30,30) = colț dreapta-sus

    Spațiul Pygame (pixeli):
        Origine: colțul STÂNGA-SUS (convenția standard în grafică 2D)
        X crește spre dreapta: 0 → WIN_W (700)
        Y crește în JOS:       0 → WIN_H (700)
        Exemplu: (0,0) = colț stânga-sus, (700,700) = colț dreapta-jos

    Diferența cheie: axa Y e INVERSATĂ între cele două sisteme.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    FORMULA DE CONVERSIE (în to_px):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    px = margin + (x / arena_w) × draw_w
         ↑ offset   ↑ normalizare [0,1]  ↑ scalare la pixeli

    py = margin + (1.0 - y / arena_h) × draw_h
                   ↑ inversare Y: y=0 → py=draw_h, y=arena_h → py=0
    """

    def __init__(self, arena_w: float, arena_h: float,
                 win_w: int, win_h: int, margin: int):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.draw_w  = win_w - 2 * margin  # lățimea zonei de desenat (fără margini)
        self.draw_h  = win_h - 2 * margin  # înălțimea zonei de desenat
        self.ox      = margin               # offset X (marginea stângă în pixeli)
        self.oy      = margin               # offset Y (marginea de sus în pixeli)

    def to_px(self, x: float, y: float):
        """
        Transformă o poziție din spațiul Python [0,ARENA_W] × [0,ARENA_H]
        în pixeli Pygame [margin, WIN_W-margin] × [margin, WIN_H-margin].

        Exemplu concret (arena 30×30, fereastră 700×700, margin 40):
            draw_w = draw_h = 620 pixeli
            to_px(0, 0)    → (40, 660)   colț stânga-jos al arenei
            to_px(30, 30)  → (660, 40)   colț dreapta-sus al arenei
            to_px(15, 15)  → (350, 350)  centrul arenei
        """
        px = self.ox + (x / self.arena_w) * self.draw_w
        py = self.oy + (1.0 - y / self.arena_h) * self.draw_h
        return int(px), int(py)

    def scale(self, units: float) -> int:
        """
        Convertește o DISTANȚĂ în unități Python la pixeli.

        Folosit pentru raze: un cerc de raza R unități Python
        are raza R × (draw_w / arena_w) pixeli.

        Exemplu: GRASS_CORE_R = 1.5u → 1.5 × (620/30) ≈ 31 pixeli
        max(1, ...) garantează că nicio rază nu e 0 pixeli (invizibil).
        """
        return max(1, int(units / self.arena_w * self.draw_w))


# ==================================================================
# CLASA Renderer — tot ce se desenează pe ecran
# ==================================================================

class Renderer:
    """
    Gestionează toate operațiile de desenare Pygame.

    ORDINEA DE DESENARE (layering, de jos în sus):
        1. screen.fill(C_BG)           ← șterge tot (fundal negru)
        2. rend.draw_arena(traps)      ← arenă + capcane
        3. rend.draw_zones(pred_pos)   ← zone radiant Fenrir (semi-transparente)
        4. rend.draw_grass(patches)    ← patch-uri iarbă
        5. rend.draw_agent(prey_pos)   ← Umbra (verde)
        6. rend.draw_agent(pred_pos)   ← Fenrir (roșu)
        7. rend.draw_hud(...)          ← statistici și legende
        8. pygame.display.flip()       ← afișează frame-ul complet

    De ce această ordine?
        Fiecare layer e desenat DEASUPRA precedentului.
        Agenții trebuie să fie vizibili deasupra zonelor și ierbii.
        HUD-ul e mereu pe deasupra a tot.
    """

    def __init__(self, cm: CoordMapper):
        self.cm      = cm
        # pygame.display.get_surface() returnează surface-ul ferestrei principale.
        # Toate operațiile de desenare se fac pe acest surface.
        self.surf    = pygame.display.get_surface()
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)  # pentru mesaje mari
        self.font_sm = pygame.font.SysFont("monospace", 13)             # pentru HUD

    def draw_arena(self, traps):
        """
        Desenează fundalul arenei și toate capcanele.

        FUNDAL ARENĂ:
            pygame.Rect(x, y, w, h) definește un dreptunghi.
            draw.rect cu thickness=0 → umplut; cu thickness=2 → doar contur.

        CAPCANE (semi-transparente):
            Problema: pygame.draw.circle nu suportă transparent direct pe screen.
            Soluția: creăm o Surface separată cu pygame.SRCALPHA (suport per-pixel alpha),
            desenăm cercul pe ea, apoi o "blit-uim" (copiem) pe screen.

            (*C_TRAP, C_TRAP_ALPHA) = unpacking tuple + adăugare alpha:
                C_TRAP = (180, 40, 40) → (*C_TRAP, 80) = (180, 40, 40, 80)
                                                                          ↑ 80/255 ≈ 31% opac

            Conturul solid (thickness=1) e desenat direct pe screen pentru claritate.
        """
        cm = self.cm
        arena_rect = pygame.Rect(cm.ox, cm.oy, cm.draw_w, cm.draw_h)
        pygame.draw.rect(self.surf, C_ARENA, arena_rect)      # umplut
        pygame.draw.rect(self.surf, C_WALL,  arena_rect, 2)   # contur 2px

        for cx, cy, r in traps:
            px, py    = cm.to_px(cx, cy)
            pr        = cm.scale(r)
            trap_surf = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            # Centrul cercului pe trap_surf e (pr, pr) — centrul suprafeței
            pygame.draw.circle(trap_surf, (*C_TRAP, C_TRAP_ALPHA), (pr, pr), pr)
            # Blit: lipim trap_surf pe screen cu colțul stânga-sus la (px-pr, py-pr)
            # astfel că centrul capcanei se aliniază cu (px, py)
            self.surf.blit(trap_surf, (px - pr, py - pr))
            # Contur solid pentru vizibilitate chiar la transparență mică
            pygame.draw.circle(self.surf, C_TRAP, (px, py), pr, 1)

    def draw_zones(self, pred_pos):
        """
        Desenează zonele radiant centrate pe Fenrir.

        Desenăm CONVECTION înainte de RADIATIVE astfel că Radiative
        (portocaliu mai intens) apare DEASUPRA Convection (galben mai pal).
        Această ordine asigură că zona mai periculoasă e vizibilă.

        Tehnica e aceeași ca la capcane: Surface cu SRCALPHA pentru transparență.
        """
        cm     = self.cm
        px, py = cm.to_px(pred_pos[0], pred_pos[1])

        for radius, color in [
            (CONVECTION_R, C_ZONE_CONV),   # galben, alpha=25 (aproape invizibil)
            (RADIATIVE_R,  C_ZONE_RAD),    # portocaliu, alpha=50 (mai vizibil)
        ]:
            pr = cm.scale(radius)
            s  = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (pr, pr), pr)
            self.surf.blit(s, (px - pr, py - pr))

    def draw_grass(self, patches: list[GrassPatch]):
        """
        Desenează toate patch-urile de iarbă.

        VIU: cerc verde (raza = GRASS_CORE_R în pixeli) cu contur mai deschis.
            max(r, 5): garantează că cercul are cel puțin 5 pixeli rază,
            chiar dacă GRASS_CORE_R mic × scale e sub 5px.

        MÂNCAT: punct mic (3px) de culoare verde-închis.
            Indică utilizatorului că acolo a existat iarbă și se regenerează.
            Util pentru a urmări comportamentul de linger al Umbrei.
        """
        for p in patches:
            px, py = self.cm.to_px(p.pos[0], p.pos[1])
            if p.alive:
                r = self.cm.scale(GRASS_CORE_R)
                pygame.draw.circle(self.surf, C_GRASS,       (px, py), max(r, 5))
                pygame.draw.circle(self.surf, (100, 255, 100),(px, py), max(r, 5), 1)
            else:
                pygame.draw.circle(self.surf, C_GRASS_EATEN, (px, py), 3)

    def draw_agent(self, pos, color, radius_px: int = 10):
        """
        Desenează un agent ca cerc colorat cu contur alb.

        Conturul alb (thickness=1) face agenții vizibili indiferent de fundalul
        pe care se află (arenă, zone, iarbă).

        radius_px este direct în pixeli (nu unități Python) —
        raza vizuală e fixă, nu depinde de dimensiunea arenei.
        Umbra: 10px, Fenrir: 12px (ușor mai mare pentru a indica dominanța).
        """
        px, py = self.cm.to_px(pos[0], pos[1])
        pygame.draw.circle(self.surf, color,          (px, py), radius_px)
        pygame.draw.circle(self.surf, (255, 255, 255),(px, py), radius_px, 1)

    def draw_hud(self, episode: int, turn: int, captures: int, timeouts: int,
                 prey_r: float, pred_r: float, last_info: str):
        """
        Desenează HUD-ul (Heads-Up Display) cu informații live.

        STÂNGA-SUS (y=8, increment 18px per linie):
            Ep XXXXX  Turn XXXX     ← episod curent și turn curent
            Capturi: X  Timeout: X  ← statistici globale
            Umbra  R: ±XX.XX        ← reward cumulativ Umbra în episodul curent
            Fenrir R: ±XX.XX        ← reward cumulativ Fenrir

        CENTRU (WIN_H // 2):
            "CAPTURAT!" (galben) sau "TIMEOUT" (albastru) — flash 300ms la end episod
            Apare doar când last_info != "" (setat în end-episod, resetat în turn următor)

        STÂNGA-JOS (WIN_H - 50):
            Legenda zonelor — culoarea textului corespunde culorii zonei pe hartă.

        font.render(text, antialias, color):
            → returnează o Surface cu textul desenat
            surf.blit(text_surface, (x, y)) → plasează textul pe ecran
        """
        lines = [
            (f"Ep {episode:>5}  Turn {turn:>4}", C_TEXT),
            (f"Capturi: {captures}  Timeout: {timeouts}", C_TEXT_DIM),
            (f"Umbra  R: {prey_r:+.2f}", C_PREY),
            (f"Fenrir R: {pred_r:+.2f}", C_PRED),
        ]
        y = 8
        for text, color in lines:
            surf = self.font_sm.render(text, True, color)
            self.surf.blit(surf, (8, y))
            y += 18

        if last_info == "captured":
            msg = self.font_lg.render("CAPTURAT!", True, C_CAPTURE)
            self.surf.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))
        elif last_info == "timeout":
            msg = self.font_lg.render("TIMEOUT", True, C_TIMEOUT)
            self.surf.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))

        y = WIN_H - 50
        for label, color in [("Convection", C_ZONE_CONV[:3]),
                              ("Radiative",  C_ZONE_RAD[:3])]:
            # C_ZONE_CONV[:3] extrage primele 3 elemente (RGB) din tuple-ul RGBA
            s = self.font_sm.render(label, True, color)
            self.surf.blit(s, (8, y))
            y += 16


# ==================================================================
# FUNCȚIA PRINCIPALĂ run()
# ==================================================================

def run(n_episodes: int = 500, seed: int = SEED,
        headless: bool = False, speed: int = 1,
        n_traps: int = N_TRAPS):
    """
    Inițializează și rulează întreaga simulare.

    PARAMETRI:
        n_episodes: câte episoade rulează simularea
        seed:       seed pentru TOATE sursele de aleatoare din simulare
                    (rng principal, rețelele SNN, l0_rng)
                    Același seed → simulare identică reproductibil
        headless:   True = fără fereastră Pygame, antrenament pur
                    (mai rapid, util pentru rulări lungi)
        speed:      câte turnuri se simulează per frame vizual
                    speed=1: real-time; speed=50: 50× mai rapid vizual
        n_traps:    ignorat intern (suprascriem cu TRAP_SIZES)
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INIȚIALIZARE SNN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # np.random.default_rng(seed): generator modern NumPy (PCG64).
    # Mai bun decât np.random.seed() pentru că e explicit și izolat.
    rng   = np.random.default_rng(seed)
    arena = Arena(width=ARENA_W, height=ARENA_H)

    # Fiecare agent are rețeaua sa SNN independentă:
    # seed diferit → greutăți inițiale diferite → agenți diverși la start
    prey_net = NeuroNet(seed=seed)
    pred_net = NeuroNet(seed=seed + 1)

    # NeuroPolicy combină rețeaua + RSTDPController + seed pentru Bernoulli spikes
    # seed+2/3: generatorul pentru secvențele stochastice de input (Bernoulli)
    # trebuie separat de rng principal pentru a nu afecta spawn-urile
    prey_policy = NeuroPolicy(prey_net, RSTDPController(prey_net), seed=seed + 2)
    pred_policy = NeuroPolicy(pred_net, RSTDPController(pred_net), seed=seed + 3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INIȚIALIZARE PYGAME
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not headless:
        pygame.init()
        pygame.display.set_caption("NeuroGame — Umbra vs Fenrir")
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        clock  = pygame.time.Clock()
        cm     = CoordMapper(ARENA_W, ARENA_H, WIN_W, WIN_H, ARENA_MARGIN)
        rend   = Renderer(cm)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LEVEL 0 — COMPORTAMENTE KANTIENE HARDCODATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # L0 = comportamente înnăscute, pre-apprise, mereu active.
    # Analogie: instinctele unui animal — nu se învață, sunt hardcodate genetic.
    #
    # De ce l0_rng SEPARAT de rng?
    # rng e folosit pentru spawn (agenți, iarbă, capcane) — determinist.
    # l0_rng e folosit pentru decizii L0 (waypoints, noise) — vrem să putem
    # schimba comportamentele L0 fără a afecta spawn-urile și invers.
    # seed+999: departe de seed, seed+1, seed+2, seed+3 pentru a evita corelații.
    l0_rng = np.random.default_rng(seed + 999)

    # State Fenrir L0:
    fenrir_waypoint      = None  # destinația curentă (np.ndarray [x,y] sau None)
    fenrir_waypoint_life = 0     # câte turnuri mai merge pe acest waypoint

    # State Umbra L0:
    umbra_waypoint       = None  # destinația curentă de foraging
    umbra_waypoint_life  = 0     # câte turnuri mai merge spre ea
    umbra_linger_timer   = 0     # câte turnuri mai lingerează după mâncat
    #   → 0 = nu lingerează; se setează la 15 când Umbra mănâncă
    umbra_linger_wp      = None  # waypoint curent de foraging local în linger
    #   → generat aleatoriu în raza LINGER_RADIUS față de locul mâncat

    # Epsilon curent (actualizat per episod în main loop)
    pred_epsilon = PRED_EPSILON_START

    # ─── Funcție helper: waypoint aleatoriu ───────────────────────
    def _random_waypoint(from_pos: np.ndarray | None = None) -> np.ndarray:
        """
        Generează un punct aleatoriu în interiorul arenei, evitând capcanele.

        Dacă from_pos e furnizat, verifică și că traiectoria from_pos→waypoint
        nu traversează nicio capcană (check_trap_segment).
        Încearcă de 200 ori; fallback la centrul arenei dacă eșuează.
        """
        for _ in range(200):
            wp = np.array([
                l0_rng.uniform(3.0, arena.width  - 3.0),
                l0_rng.uniform(3.0, arena.height - 3.0),
            ])
            if arena.check_trap(wp):
                continue   # waypoint-ul însuși e pe capcană
            if from_pos is not None and arena.check_trap_segment(from_pos, wp):
                continue   # traiectoria traversează o capcană
            return wp
        # Fallback: centrul arenei (puțin probabil să fie pe capcană)
        return np.array([arena.width / 2, arena.height / 2])

    # ─── Funcție helper: acțiune spre o țintă ─────────────────────
    def _action_toward(pos: np.ndarray, target: np.ndarray, noise: int = 3) -> int:
        """
        Calculează acțiunea (0-31) care mișcă agentul de la pos spre target.

        SPAȚIUL DE ACȚIUNI:
            32 direcții uniforme pe un cerc complet (360°/32 = 11.25° per bucket).
            Acțiunea 0 = Est (0°), 8 = Nord (90°), 16 = Vest (180°), 24 = Sud (270°).

        FORMULA:
            delta = target - pos          → vectorul direcție [dx, dy]
            angle = atan2(dy, dx)         → unghiul în radiani [-π, π]
            base  = angle / (2π) × 32     → bucket-ul corespunzător [0, 31]
            off   = l0_rng.integers(-noise, noise+1)  → offset aleatoriu

            (base + off) % 32             → acțiunea finală (circular modulo)

        PARAMETRUL noise:
            noise=1:  ±1 bucket = ±11.25° → direcție aproape exactă (viziune)
            noise=4:  ±4 bucket = ±45°    → direcție aproximativă (miros mediu)
            noise=12: ±12 bucket = ±135°  → direcție foarte vagă (miros slab)

            l0_rng.integers(-noise, noise+1):
                ← generează un int uniform din [-noise, noise] INCLUSIV
                ← noise+1 pentru că integers e [low, high) = exclusiv dreapta
        """
        delta = target - pos
        angle = math.atan2(delta[1], delta[0])
        base  = int(angle / (2 * math.pi) * 32) % 32
        off   = int(l0_rng.integers(-noise, noise + 1))
        return (base + off) % 32

    # ─── Politica L0 Fenrir: patrol + urmărire ────────────────────
    def pred_policy_l0(state_dict: dict) -> int:
        """
        Politica Level 0 a lui Fenrir.

        state_dict conține (furnizat de GameEngine per turn):
            "self_pos":     np.ndarray [x, y] — poziția lui Fenrir
            "opponent_pos": np.ndarray [x, y] — poziția Umbrei
            "arena":        obiectul Arena
            "stamina":      float (stamina Fenrir, dacă există)

        DECIZIE (prioritate de sus în jos):

        1. Umbra e în RADIATIVE_R (< 5u)?
           → URMĂRIRE ACTIVĂ: apelează SNN (pred_policy) cu 10% random
           → 10% random previne fixarea pe direcție greșită ("overfitting" local)
           → SNN primește state_dict și returnează acțiunea optimă învățată

        2. PATROL: mergi spre un waypoint random în arenă
           → Waypoint nou dacă: niciun waypoint (None), viața a expirat (≤0),
             sau am ajuns la destinație (distanță < 1.5u)
           → Viața waypoint: 20-50 turnuri (aleatoriu)
           → noise=3 în _action_toward → Fenrir nu merge perfect drept

        nonlocal: permite funcției interne să MODIFICE variabilele din run()
        (Python: fără nonlocal, variabila ar fi locală și modificările s-ar pierde)
        """
        nonlocal fenrir_waypoint, fenrir_waypoint_life, pred_epsilon

        pos  = state_dict["self_pos"]
        opp  = state_dict["opponent_pos"]
        dist = float(np.linalg.norm(pos - opp))

        if dist < RADIATIVE_R:
            # Umbra e aproape — SNN decide urmărirea
            if l0_rng.random() < 0.1:
                return int(l0_rng.integers(0, 32))
            return pred_policy(state_dict)

        # Patrol: waypoint nou dacă necesar
        if fenrir_waypoint is None or fenrir_waypoint_life <= 0:
            fenrir_waypoint      = _random_waypoint(from_pos=pos)
            fenrir_waypoint_life = int(l0_rng.integers(20, 50))

        fenrir_waypoint_life -= 1

        if float(np.linalg.norm(pos - fenrir_waypoint)) < 1.5:
            fenrir_waypoint      = _random_waypoint(from_pos=pos)
            fenrir_waypoint_life = int(l0_rng.integers(20, 50))

        return _action_toward(pos, fenrir_waypoint)

    # ─── Politica L0 Umbra: sensing iarbă + foraging + linger ─────
    def prey_policy_l0(state_dict: dict) -> int:
        """
        Politica Level 0 a Umbrei — cea mai complexă din simulare.

        INJECȚIE GRASS:
            state_dict["grass_patches"] = grass_ref["patches"]
            Adăugăm manual lista de patch-uri active în state_dict.
            De ce? SNN encoding (encoding.py) citește "grass_patches" pentru
            a genera cei 8 neuroni direcționali (grass_ring).
            grass_ref e actualizat în main loop după fiecare turn.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        IERARHIA DE DECIZII (cea mai prioritară prima):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        1. PERICOL (dist_fenrir < CONVECTION_R = 10u):
           → SNN preia controlul COMPLET
           → L0 nu intervine — rețeaua decide fuga
           → Motivație: sub 10u, Fenrir e periculos și SNN a învățat
             strategii de evadare mai bune decât waypointul L0

        2. LINGER (umbra_linger_timer > 0):
           → Umbra a mâncat recent (15 turnuri în urmă sau mai puțin)
           → Se întoarce ușor spre locul ierbii mâncate (noise=8)
           → Biologic: sațietatea → relaxare → vulnerabilitate
           → Mecanic: creează fereastra de vânătoare pentru Fenrir

        3. SENSING IARBĂ (gradient de acuratețe):

           Găsim cel mai apropiat patch activ:
               dists_g = [||pos - patch_i||  for patch_i in active]
               min_idx = argmin(dists_g)      ← indexul celui mai apropiat
               nearest = active[min_idx]      ← coordonatele lui
               d_grass = dists_g[min_idx]     ← distanța până la el

           Sub-zone (distanță față de nearest):

           a) CORE (< 1.5u): Umbra e PE iarbă — MÂNCAT PROGRESIV
              → Rămâne în zonă (noise=2 = mișcare mică, nu pleacă)
              → GrassManager.update() incrementează eating_progress per turn
              → După GRASS_EAT_TURNS (3) turnuri consecutive: patch consumat
              → Main loop setează umbra_linger_timer=15 și umbra_waypoint
              → Dacă Umbra pleacă înainte → eating_progress se resetează (întrerupt)

           b) RADIATIVE (< 4.0u): Umbra vede clar iarba
              → Direcție precisă (noise=1 = ±11°)
              → Analogie biologică: viziune directă

           c) CONVECTION (< 8.0u): Umbra simte miros incert
              → 50% șansă de direcție corectă (noise=4)
              → 50% acțiune complet random (dezorientată)
              → Modelează olfacția incertă la distanță medie

           d) CORONA (< 15.0u): Umbra simte miros vag
              → Direcție cu zgomot mare (noise=12 = ±135°)
              → Știe că e iarbă pe undeva, nu știe exact unde
              → Modelează detectarea difuză de molecule olfactive

           e) DINCOLO DE 15u: Umbra nu simte nimic
              → Nu intră în nici un if → cade la Foraging (waypoint)

        4. FORAGING (nicio iarbă detectată sau prea departe):
           → Waypoint aleatoriu în arenă (25-60 turnuri viață)
           → Comportament de bază: explorare teritoriu
        """
        nonlocal umbra_waypoint, umbra_waypoint_life, umbra_linger_timer, umbra_linger_wp

        # Injectăm grass în state_dict pentru SNN encoding
        state_dict["grass_patches"] = grass_ref["patches"]

        pos  = state_dict["self_pos"]
        opp  = state_dict["opponent_pos"]
        dist = float(np.linalg.norm(pos - opp))

        # ── 1. PERICOL ────────────────────────────────────────────
        if dist < CONVECTION_R:
            return prey_policy(state_dict)

        # ── 2. LINGER ─────────────────────────────────────────────
        # Umbra a mâncat recent → caută mai multă iarbă în zonă (nu stă pe loc).
        # Generăm waypoints aleatoare într-o rază de LINGER_RADIUS în jurul
        # locului mâncat — comportament de foraging local, ca un animal care
        # știe că zona e bogată și continuă să pască.
        if umbra_linger_timer > 0:
            umbra_linger_timer -= 1
            linger_center = umbra_waypoint if umbra_waypoint is not None else pos
            # Dacă am ajuns la waypoint-ul de linger sau nu avem unul → nou
            if (umbra_linger_wp is None or
                    float(np.linalg.norm(pos - umbra_linger_wp)) < 1.0):
                LINGER_RADIUS = 5.0
                for _ in range(50):
                    angle = l0_rng.uniform(0, 2 * np.pi)
                    r     = l0_rng.uniform(1.0, LINGER_RADIUS)
                    wp    = linger_center + np.array([np.cos(angle) * r,
                                                      np.sin(angle) * r])
                    # Clampăm în interiorul arenei
                    wp[0] = float(np.clip(wp[0], 2.0, arena.width  - 2.0))
                    wp[1] = float(np.clip(wp[1], 2.0, arena.height - 2.0))
                    if not arena.check_trap(wp):
                        umbra_linger_wp = wp
                        break
                else:
                    umbra_linger_wp = linger_center.copy()
            return _action_toward(pos, umbra_linger_wp, noise=3)

        # ── 3. SENSING IARBĂ ──────────────────────────────────────
        GRASS_SENSE_R = 15.0
        active = grass_ref["patches"]
        if active:
            dists_g = [float(np.linalg.norm(pos - np.array(p))) for p in active]
            min_idx = int(np.argmin(dists_g))
            nearest = np.array(active[min_idx])
            d_grass = dists_g[min_idx]

            if d_grass < GRASS_CORE_R:
                # Umbra e în CORE — rămâne pe loc (mâncat progresiv, GRASS_EAT_TURNS turnuri).
                # Linger se setează din main loop după ce GrassManager confirmă consumarea.
                return _action_toward(pos, nearest, noise=2)

            elif d_grass < GRASS_RAD_R:
                return _action_toward(pos, nearest, noise=1)

            elif d_grass < GRASS_CONV_R:
                if l0_rng.random() < 0.5:
                    return _action_toward(pos, nearest, noise=4)
                return int(l0_rng.integers(0, 32))

            elif d_grass < GRASS_SENSE_R:
                return _action_toward(pos, nearest, noise=12)

        # ── 4. FORAGING ───────────────────────────────────────────
        if umbra_waypoint is None or umbra_waypoint_life <= 0:
            umbra_waypoint      = _random_waypoint(from_pos=pos)
            umbra_waypoint_life = int(l0_rng.integers(25, 60))

        umbra_waypoint_life -= 1

        if float(np.linalg.norm(pos - umbra_waypoint)) < 1.5:
            umbra_waypoint      = _random_waypoint(from_pos=pos)
            umbra_waypoint_life = int(l0_rng.integers(25, 60))

        return _action_toward(pos, umbra_waypoint)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GRASS MANAGER + EPSILON-GREEDY WRAPPER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    grass_mgr = GrassManager(N_GRASS, arena, rng)

    pred_pos_hist: list = []
    # Rezervat pentru viitor: anti-stuck detection.
    # Dacă Fenrir stă în același loc > N turnuri → boost epsilon temporar.
    # Momentan: doar inițializat și resetat per episod.

    def pred_policy_eps(state_dict: dict) -> int:
        """
        Wrapper epsilon-greedy GLOBAL peste pred_policy_l0.

        Cu probabilitate pred_epsilon → acțiune random (orice direcție)
        Altfel → pred_policy_l0 (patrol sau urmărire SNN)

        DE CE DOUĂ NIVELURI DE EPSILON?
            pred_policy_l0 are deja 10% random când urmărește (dist < RADIATIVE_R).
            pred_policy_eps adaugă explorare GLOBALĂ — inclusiv când Fenrir e departe.

            Efectul combinat:
            - Dist > RADIATIVE_R: pred_epsilon random + (1-pred_epsilon) × patrol L0
            - Dist < RADIATIVE_R: pred_epsilon random + (1-pred_epsilon) × (10% random + 90% SNN)

        l0_rng.random() generează un float uniform [0, 1).
        Dacă e < pred_epsilon (ex: 0.4) → condiția e adevărată 40% din timp.
        """
        if l0_rng.random() < pred_epsilon:
            return int(l0_rng.integers(0, 32))
        return pred_policy_l0(state_dict)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STATISTICI GLOBALE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    captures  = 0   # numărul total de episoade câștigate de Fenrir (prin captură)
    timeouts  = 0   # numărul total de episoade terminate prin timeout
    last_info = ""  # "captured" / "timeout" / "trap" — rezultatul ultimului episod

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CAPCANE FIXE — spawned o singură dată, rămân toată sesiunea
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # DE CE FIXE și nu regenerate per episod?
    #
    # Dacă capcanele se mișcă → agenții nu pot forma o asociere
    # spațială specifică ("colțul ăsta e periculos").
    # Memoria spațială necesită stimul STABIL în timp.
    #
    # Biologică: o mlaștină sau un râu rămân în același loc.
    # Un animal care supraviețuiește le evită din memorie, nu din ghicire.
    #
    # Tehnic: pentru _ in range(500) + break → încearcă max 500 poziții,
    # ia prima validă (break imediat după append).
    # Loop-ul interior e o construcție de "try until success" cu 1 succes garantat.
    arena.traps = []
    for r in TRAP_SIZES:
        for _ in range(500):
            cx = rng.uniform(3.0, arena.width  - 3.0)
            cy = rng.uniform(3.0, arena.height - 3.0)
            arena.traps.append((cx, cy, r))
            break   # ← ieșim după primul succes; capcanele nu se verifică față de iarbă

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PLATEAU DETECTOR — monitorizare consolidare SNN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    episode_history: list[tuple[float, float, str]] = []
    # Acumulăm (ep_prey_r, ep_pred_r, info) pentru fiecare episod.
    # info = "captured" / "timeout" / "trap" — rezultatul episodului.
    # Folosit de plateau detector pentru capture rate și timeout growth.

    _w_in_snapshot = prey_policy.net.W_in.copy()
    # Snapshot al matricei de greutăți W_in a Umbrei (shape: 256×73).
    # .copy() este ESENȚIAL: fără copy(), _w_in_snapshot ar fi o REFERINȚĂ
    # la același array — s-ar modifica odată cu W_in și delta ar fi mereu 0.

    _plateau_hits = 0
    # Numărul de check-uri CONSECUTIVE sub prag.
    # Resetat la 0 de fiecare dată când un check depășește pragul.

    PLATEAU_WINDOW    = 100   # fereastră de analiză: comparăm 100 ep vs 100 ep
    PLATEAU_W_RATE    = 1e-2  # ΔW_in relativă < 1% per 100 ep
    # ↑ Recalibrat din date reale (ep50-8000): ΔW_in oscilează 0.007-0.015
    # indefinit cu LR=0.01. Pragul 0.002 era neatins — 0.01 e plafonul inferior real.
    PLATEAU_CAP_DELTA = 0.04  # delta capture rate < 4pp între ferestre consecutive
    # ↑ Înlocuiește formula Δreward care ceda la schimbare de semn (max 2.0 matematic)
    PLATEAU_TOUT_MIN  = 10    # minim 10 timeouts în fereastra nouă (Umbra supraviețuiește)

    def _check_plateau(ep: int) -> None:
        """
        Verifică periodic dacă SNN-ul a atins platoul de consolidare.
        Apelat la fiecare PLATEAU_WINDOW (100) episoade.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        METRICA 1: ΔW_in — schimbarea greutăților (recalibrat)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        w_delta = ||W_in_acum - W_in_snapshot|| / ||W_in_snapshot||  (norma Frobenius)

        Pragul e 0.01 (1%), recalibrat din date reale (ep50-8000):
            - La LR=0.01 constant, ΔW_in oscilează 0.007-0.015 INDEFINIT.
            - Pragul original 0.002 era NEATINS matematic → fals negativ permanent.
            - 0.01 = plafonul inferior al oscilației = semnal că learning e minim.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        METRICA 2: cap_delta — stabilitatea capture rate (înlocuiește rew_frac)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        cap_rate_new = capturi% din ultimele 100 ep
        cap_rate_old = capturi% din ep 100-200 în urmă
        cap_delta    = |cap_rate_new - cap_rate_old|  ∈ [0, 1]

        De ce nu mai folosim rew_frac?
            Formulă veche: rew_frac = |mean_new - mean_old| / denom
            BUG: când mean_old < 0 și mean_new > 0 (schimbare de semn):
                denom = (|neg| + |pos|) / 2 → mic
                numărătorul = |pos - neg| → mare
                rew_frac → 2.0 (plafonul matematic) = FALS POZITIV permanent

        Capture rate e [0,1], niciodată negativă → fără bug de semn.
        Prag 0.04 = 4 puncte procentuale delta între ferestre = stabilizare comportamentală.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        METRICA 3: tout_new — timeouts în fereastra nouă (semnal primar)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Conform datelor reale: PRIMUL semnal de consolidare au fost timeouts-urile.
        La ep~3550 au apărut primele timeouts → Umbra a „învățat" să supraviețuiască 500 de turnuri.
        Fără timeouts = Umbra moare rapid → SNN nu a consolidat reflexul de evitare.

        Prag: minim 10 timeouts în fereastra de 100 ep = ~10% episoade cu supraviețuire completă.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CONDIȚIE PLATOU: toate trei metrici satisfăcute simultan
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        is_plateau = w_delta < 0.01 AND cap_delta < 0.04 AND tout_new >= 10

        2 check-uri consecutive → CONSOLIDAT (anti-zgomot: un platou accidental nu declanșează)
        Orice check care eșuează resetează _plateau_hits la 0.
        """
        nonlocal _w_in_snapshot, _plateau_hits

        if ep < PLATEAU_WINDOW * 2:
            return   # nevoie de minim 200 episoade pentru 2 ferestre complete

        # ── Metrica 1: ΔW_in ────────────────────────────────────────
        w_now   = prey_policy.net.W_in
        w_delta = float(np.linalg.norm(w_now - _w_in_snapshot)) / \
                  (float(np.linalg.norm(_w_in_snapshot)) + 1e-9)
        _w_in_snapshot = w_now.copy()   # snapshot pentru check-ul următor

        # ── Metrica 2: capture rate delta ───────────────────────────
        win_new = episode_history[-PLATEAU_WINDOW:]
        win_old = episode_history[-PLATEAU_WINDOW * 2:-PLATEAU_WINDOW]
        cap_rate_new = sum(1 for e in win_new if e[2] == "captured") / PLATEAU_WINDOW
        cap_rate_old = sum(1 for e in win_old if e[2] == "captured") / PLATEAU_WINDOW
        cap_delta    = abs(cap_rate_new - cap_rate_old)

        # ── Metrica 3: timeouts în fereastra nouă ───────────────────
        tout_new = sum(1 for e in win_new if e[2] == "timeout")

        is_plateau = (w_delta   < PLATEAU_W_RATE    and
                      cap_delta < PLATEAU_CAP_DELTA  and
                      tout_new  >= PLATEAU_TOUT_MIN)

        print(f"\n[CONSOLIDARE Ep {ep:5d}]"
              f"  ΔW_in={w_delta:.5f}(prag {PLATEAU_W_RATE})"
              f"  Δcap={cap_delta:.3f}(prag {PLATEAU_CAP_DELTA})"
              f"  tout/100={tout_new}(min {PLATEAU_TOUT_MIN})"
              f"  cap%={cap_rate_new*100:.1f}")

        if is_plateau:
            _plateau_hits += 1
            print(f"  → PLATOU #{_plateau_hits}/2 — reflexele se stabilizează")
            if _plateau_hits >= 2:
                print(f"\n{'═'*55}")
                print(f"  CREIER MUSCĂ CONSOLIDAT — episod {ep}")
                print(f"  ΔW_in stabil, capture rate stabilă, timeouts prezente.")
                print(f"  Următor pas: checkpoint + proto-hippocampus.")
                print(f"{'═'*55}\n")
        else:
            _plateau_hits = 0
            print(f"  → Încă în formare.")

    # ==================================================================
    # LOOP PRINCIPAL — EPISOADE
    # ==================================================================

    for ep in range(1, n_episodes + 1):

        # ─── Reset la început de episod ───────────────────────────
        # random_spawn: generează poziții inițiale pentru ambii agenți
        # cu distanța minimă garantată (8u) și fără overlap cu capcanele
        prey_pos, pred_pos = random_spawn(arena, rng)
        prey = Prey(prey_pos)
        pred = Predator(pred_pos)

        # reset_episode():
        #   net.reset() → potențialele LIF ale tuturor neuronilor → 0
        #   rstdp.reset() → trace-urile STDP → 0
        # Important: greutățile W_in/W_rec/W_out NU se resetează — persistă între episoade
        prey_policy.reset_episode()
        pred_policy.reset_episode()

        grass_mgr.reset()        # iarbă nouă în poziții noi pentru fiecare episod
        umbra_linger_timer = 0   # resetăm linger-ul (Umbra nu a mâncat la start)
        umbra_linger_wp    = None
        pred_pos_hist.clear()

        # grass_ref: dict mutable shared între main loop și prey_policy_l0
        # De ce dict și nu variabilă simplă?
        # prey_policy_l0 e o funcție closure — poate citi variabile din run()
        # dar nu le poate MODIFICA (fără nonlocal). Un dict e mutable,
        # deci prey_policy_l0 poate citi grass_ref["patches"] fără nonlocal.
        # Main loop actualizează grass_ref["patches"] după fiecare turn.
        grass_ref = {"patches": grass_mgr.active_positions()}

        # GameEngine gestionează fizica: mișcare, coliziuni cu pereți și capcane,
        # condiția de captură, timeout-ul.
        # Primește cele două politici și le apelează pe rând per turn.
        engine = GameEngine(
            arena=arena, prey=prey, predator=pred,
            prey_policy=prey_policy_l0,   # L0 cu grass sensing + apel intern SNN
            pred_policy=pred_policy_eps,  # epsilon-greedy wrapper peste L0
        )

        ep_prey_r = 0.0   # acumulator reward Umbra (resetat per episod)
        ep_pred_r = 0.0   # acumulator reward Fenrir

        # ─── Loop turnuri ─────────────────────────────────────────
        # engine.done devine True când: captură / capcană / timeout
        while not engine.done:

            # EVENIMENTE PYGAME (procesate o dată per frame, nu per turn)
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            speed = min(speed + 1, 50)   # accelerare (+)
                        if event.key == pygame.K_MINUS:
                            speed = max(speed - 1, 1)    # decelerare (-)

            # SIMULĂM `speed` TURNURI PER FRAME
            # La speed=1:  un turn per frame  → vizualizare fluidă
            # La speed=10: 10 turnuri pe frame → antrenament de 10× mai rapid vizual
            # La speed=50: 50 turnuri pe frame → aproape headless cu fereastră
            for _ in range(speed):
                if engine.done:
                    break   # nu mai simulăm turnuri după ce episodul s-a terminat

                # engine.step():
                #   1. Apelează prey_policy_l0(state_dict) → acțiune Umbra (0-31)
                #   2. Apelează pred_policy_eps(state_dict) → acțiune Fenrir (0-31)
                #   3. Mișcă ambii agenți în direcțiile alese
                #   4. Clipuieste pozițiile la marginile arenei
                #   5. Verifică captură (dist < CORE_R) și capcane
                #   6. Returnează EpisodeResult cu done, info, pozițiile noi
                result = engine.step()

                # Update iarbă DUPĂ mișcarea agenților
                # eaten_pos ≠ None → Umbra tocmai a terminat de mâncat un patch (după GRASS_EAT_TURNS turnuri)
                grass_reward, eaten_pos = grass_mgr.update(engine.prey.pos)
                if eaten_pos is not None:
                    # Patch consumat → activăm linger: Umbra rămâne lângă locul mâncării
                    umbra_linger_timer = 15
                    umbra_waypoint     = eaten_pos
                # Actualizăm lista de patch-uri pentru turn-ul următor
                # (prey_policy_l0 va citi grass_ref["patches"] în turn-ul următor)
                grass_ref["patches"] = grass_mgr.active_positions()

                # Reward radiant bazat pe distanța curentă predator-prey
                prey_r, pred_r = compute_rewards(
                    engine.prey.pos,
                    engine.predator.pos,
                    result.done and result.info == "captured",
                    result.done and result.info == "trap",
                )
                # Adăugăm motivația de iarbă la reward-ul Umbrei
                # grass_reward = 0.0 tipic, 0.05/0.3 aproape, 5.0 la mâncat
                prey_r += grass_reward

                # apply_reward(): transmite reward-ul la RSTDPController
                # RSTDPController aplică regula STDP scalată cu reward-ul:
                #   ΔW = η × reward × (pre_trace × post_spike - post_trace × pre_spike)
                # Greutățile W_in și W_rec se modifică după fiecare turn.
                prey_policy.apply_reward(prey_r)
                pred_policy.apply_reward(pred_r)

                ep_prey_r += prey_r
                ep_pred_r += pred_r

            # RENDER (o dată per frame, după toate turnurile din speed)
            if not headless:
                screen.fill(C_BG)                                    # 1. fundal
                rend.draw_arena(arena.traps)                         # 2. arenă + capcane
                rend.draw_zones(engine.predator.pos)                 # 3. zone radiant
                rend.draw_grass(grass_mgr.patches)                   # 4. iarbă
                rend.draw_agent(engine.prey.pos,     C_PREY, 10)     # 5. Umbra
                rend.draw_agent(engine.predator.pos, C_PRED, 12)     # 6. Fenrir
                rend.draw_hud(ep, engine.turn, captures, timeouts,
                              ep_prey_r, ep_pred_r, "")              # 7. HUD
                pygame.display.flip()   # schimbă buffer-ul → afișează frame-ul
                clock.tick(FPS_CAP)     # limitează la 60 FPS

        # ─── End episod ───────────────────────────────────────────
        last_info = result.info   # "captured", "timeout", sau "trap"
        if result.info == "captured":
            captures += 1
        elif result.info == "timeout":
            timeouts += 1

        if not headless:
            # Flash de 300ms cu rezultatul episodului
            screen.fill(C_BG)
            rend.draw_arena(arena.traps)
            rend.draw_agent(engine.prey.pos,     C_PREY, 10)
            rend.draw_agent(engine.predator.pos, C_PRED, 12)
            rend.draw_hud(ep, engine.turn, captures, timeouts,
                          ep_prey_r, ep_pred_r, last_info)
            pygame.display.flip()
            pygame.time.wait(300)   # 300 millisecunde pauză

        # Adăugăm datele episodului la istoric (pentru plateau detector)
        # last_info = "captured" / "timeout" / "trap" — rezultatul episodului
        episode_history.append((ep_prey_r, ep_pred_r, last_info))

        # Decrementăm epsilon: ε_nou = max(MIN, ε_vechi × DECAY)
        # Exemplu: ep=100, ε = 0.4 × 0.999^100 ≈ 0.361
        pred_epsilon = max(PRED_EPSILON_MIN, pred_epsilon * PRED_EPSILON_DECAY)

        # Statistici la fiecare 50 episoade
        if ep % 50 == 0:
            rate = captures / ep * 100
            print(f"Ep {ep:5d} | captures {captures:4d} ({rate:.1f}%) "
                  f"| timeouts {timeouts:4d} | ε={pred_epsilon:.3f} | speed x{speed}")

        # Verificare platou la fiecare 100 episoade
        if ep % PLATEAU_WINDOW == 0:
            _check_plateau(ep)

    if not headless:
        pygame.quit()

    print(f"\nFinal: {captures} capturi / {n_episodes} episoade "
          f"({captures/n_episodes*100:.1f}%)")


# ==================================================================
# ENTRYPOINT
# ==================================================================

def parse_args():
    p = argparse.ArgumentParser(description="NeuroGame Visualizer")
    p.add_argument("--episodes", type=int,  default=500)
    p.add_argument("--seed",     type=int,  default=SEED)
    p.add_argument("--speed",    type=int,  default=1,
                   help="turnuri per frame (+ / - pentru ajustare live)")
    p.add_argument("--traps",    type=int,  default=N_TRAPS)
    p.add_argument("--headless", action="store_true",
                   help="training fără fereastră Pygame")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        n_episodes=args.episodes,
        seed=args.seed,
        headless=args.headless,
        speed=args.speed,
        n_traps=args.traps,
    )
