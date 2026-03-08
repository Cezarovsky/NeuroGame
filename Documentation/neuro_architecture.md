# NeuroGame — Neuromorphic Architecture

**Status:** Implementat și testat (March 2026)  
**Backend:** NumPy LIF (fallback Loihi2/LAVA când hardware disponibil)  
**Scop:** Studiu academic — co-evoluție adversarială cu R-STDP

---

## 1. Contextul conceptual

### Problema
Doi agenți autonomi — pradă (iepure) și prădător (lup) — învață simultan într-un mediu adversarial fără supervizare explicită. Nu există „jucător corect" definit: fiecare agent descoperă strategia prin experiența sa proprie.

### De ce neuromorphic?
- **Codificare temporală**: Spiking Neural Networks (SNN) encodează informație în *timing-ul* spike-urilor, nu în valori continue. Biologic mai fidel față de sisteme senzoriale reale (retine silicon, cochlee artificiale).
- **R-STDP**: STDP pur (Spike-Timing Dependent Plasticity) nu poate distinge rezultate bune de rele — ambele schimbă greutățile. R-STDP adaugă un semnal neuromodulator (analog dopaminei) care *modulează* plasticitatea: aceeași secvență de spike-uri produce LTP sau LTD în funcție de reward.
- **Recurrente sparse**: Memoria temporală implicită fără arhitecturi recurrente explicite (LSTM/GRU). Starea precedentă se propagă prin conexiunile recurrente → agenții pot „anticipa" fără un modul de memorie separat.

### Referințe cheie
- Izhikevich (2007) — *Solving the distal reward problem through linkage of STDP and dopamine signaling*
- Rao & Ballard (1999) — greutăți W_out fixe (predictor coding)
- Intel Loihi 2 / LAVA framework — simulare neuromorphică pe hardware dedicat

---

## 2. Arhitectura rețelei

```
Input (65 LIF)
    │
    │  W_in (256 × 65)
    ▼
Hidden (256 LIF)  ◄──── W_rec (256 × 256, sparse 20%) ────┐
    │                                                       │
    └──────────────────────────────────────────────────────┘
    │
    │  W_out (32 × 256)  [fix — nu e actualizat de R-STDP]
    ▼
Output (32 LIF)
    │
    ▼
Winner-Take-All (ultimii 20 timesteps din 100)
    │
    ▼
Action index (0–31)
```

### Parametri LIF (Leaky Integrate-and-Fire)

| Parametru | Valoare | Semnificație |
|-----------|---------|--------------|
| `τ_mem` | 10 timesteps | Constanta de timp a membranei |
| `V_thresh` | 1.0 | Prag de spike |
| `V_reset` | 0.0 | Potențial după spike |
| Rezoluție turn | 100 timesteps | 1 turnul jocului = 100 iterații LIF |

### Inițializare greutăți

| Matrice | Shape | Metodă |
|---------|-------|--------|
| `W_in` | (256, 65) | Xavier uniform |
| `W_rec` | (256, 256) | Sparse 20%, Xavier × 0.5, fără autoconexiuni |
| `W_out` | (32, 256) | Xavier uniform, **fix** (nu se antrenează) |

`W_rec_mask` (matrice booleană) este reținut permanent — R-STDP actualizează doar conexiunile existente la inițializare. Sparsitatea rămâne constantă pe durata antrenamentului.

---

## 3. Encodarea stării (65 neuroni input)

### 3.1 Inelul oponentului (neuroni 0–31)

32 de neuroni dispuși pe un inel circular (0°–360°, 11.25°/bucket).

Activare **gaussiană** centrată pe direcția spre oponent:
```
amplitude = min(1.0, (DIST_SCALE / dist)²)     # mai aproape = mai puternic
rates[i] = amplitude × exp(-0.5 × (d_angular / σ)²)
```
unde `σ = 2 bucket-uri ≈ 22.5°` și `DIST_SCALE = 10.0`.

**Efectul biologic simulat**: Câmpuri receptive circulare cu inhibiție laterală — un neuron e maxim activ când oponentul e exact în direcția lui, cu scădere graduală pe ambele laturi.

### 3.2 Inelul pereților (neuroni 32–63)

32 de neuroni, câte unul per direcție (aceleași 32 de unghiuri ca inelul oponentului).

```
proximity[i] = 1.0 - clip(dist_to_wall(angle_i) / max_arena_diagonal, 0, 1)
```

Neuron activ = perete aproape în acea direcție. Agentul **nu e informat explicit** că pereții sunt periculos — riscul e learned emergent (morți la colțuri generează rewards negative → R-STDP deprimă weight-urile asociate).

### 3.3 Stamina ratio (neuronul 64)

```
rate = stamina / stamina_max    ∈ [0.0, 1.0]
```

Un singur neuron care codifică cât „combustibil" mai are prada. Predatorul primește `rate = 0` la acest neuron (nu are concept de stamina).

---

## 4. Spațiul de acțiuni

### 4.1 Prey — arc 180°

32 de acțiuni = 32 de puncte pe un arc semicircular perpendicular pe vectorul `predator → prey`.

```
flee_angle = atan2(prey_y - pred_y, prey_x - pred_x)
action_angles = flee_angle + linspace(-π/2, +π/2, 32)
candidates[i] = prey_pos + max_step(turn) × [cos(angle[i]), sin(angle[i])]
```

**Raza arcului** = `stamina / log(turn + 2)` — distanța maximă scade logaritmic pe măsură ce stamina se epuizează și turnu avansează.

Arc 180° (nu 360°): prada *nu poate* fugi direct spre prădător — spațiul de acțiuni e constrained biologic (un iepure nu aleargă spre lup).

### 4.2 Predator — 360° intercept-ponderat

32 de acțiuni = 32 de puncte uniform pe cerc complet, raza fixă `speed = 1.5`.

```
intercept_target = prey_pos + prey_velocity_avg × 2_turns
action_positions = pred_pos + speed × [cos(angle_i), sin(angle_i)]  for i in 0..31
```

Predatorul calculează unde **va fi** prada (intercept), nu unde este (pursuit). Previne degenerarea în mișcare liniară colinearizată — problema clasică a pursuit-curve care crează agenți prea predictibili.

Viteza medie a prăzii e estimată din ultimele `VELOCITY_WINDOW = 4` turnuri.

---

## 5. R-STDP — Reward-modulated STDP

### 5.1 Trace-uri STDP (per timestep)

```
x_pre[t]  = x_pre[t-1] × (1 - 1/τ_pre)  + spike_pre[t]
x_post[t] = x_post[t-1] × (1 - 1/τ_post) + spike_post[t]

e[t] += A_plus  × outer(spike_post, x_pre)    # LTP: post după pre
      - A_minus × outer(x_post, spike_pre)    # LTD: pre după post
```

**Parametri:**

| Parametru | Valoare |
|-----------|---------|
| `τ_pre = τ_post` | 20 timesteps |
| `A_plus = A_minus` | 0.005 (simetric) |

### 5.2 Eligibility trace per turn

La sfârșitul fiecărui turn (100 timesteps), eligibility trace e normalizată:
```
elig_turn = mean(e[t] for t in 0..99)
```
și adăugată într-o fereastră circulară de `REWARD_WINDOW = 3` turnuri.

### 5.3 Actualizare greutăți la reward

```
ΔW = LR × reward × Σ(k=0..2: REWARD_DECAY^k × elig_window[-k])
```

**Parametri:**

| Parametru | Valoare |
|-----------|---------|
| `REWARD_WINDOW` | 3 turnuri |
| `REWARD_DECAY` | 0.7 (discount exponențial per turn) |
| `LR` | 0.01 |
| `W_MIN / W_MAX` | -1.0 / +1.0 (clipping după update) |

### 5.4 Semnalul de reward

| Situație | Prey reward | Predator reward |
|----------|-------------|-----------------|
| Supraviețuire (per turn) | +0.1 | -0.05 |
| Captură | **-10.0** | **+10.0** |
| Timeout (>500 turnuri) | +0.1 (last turn) | -0.05 (last turn) |

Reward asimetric deliberat: predatorul plătește un cost continuu de „vânătoare" (presiune evolutivă să fie eficient), prada primește bonus per supraviețuire (presiune să evite capturat, nu să stea pe loc).

---

## 6. Stamina (Prey)

```
max_step(turn) = stamina × SPEED_SCALE / log(turn + 2)
```

| Parametru | Valoare |
|-----------|---------|
| `STAMINA_MAX` | 30.0 |
| `STAMINA_REGEN` | +5.0 la fiecare 3 turnuri |
| `REGEN_EVERY` | 3 turnuri |
| `SPEED_SCALE` | 1.0 |

**Efectul designului**: Prada are burst de viteză la început (stamina plină), dar e forțată să gestioneze energia. Regenerarea periodică crează cicluri naturale de „risc/recuperare".

---

## 7. Memoria temporală — recurrente sparse

Spre deosebire de LSTM/GRU care au un modul explicit de memorie, recurrentele sparse din W_rec transmit activarea anterioară ca input suplimentar:

```
I_hidden[t] = W_in @ x[t] + W_rec @ h[t-1]
```

**Efectul**: Rețeaua poate reține context din timestep-urile anterioare ale aceluiași turn *și* din turnurile anterioare (starea nu e resetată între turnuri în cadrul unui episod).

**Jinking emergent**: Prada nu e programată să schimbe direcția impredictibil. Schimbările de direcție apar din interacțiunea:
- Recurrente care rețin traiectoria anterioară
- R-STDP care recompensează supraviețuirea (orice strategie funcționează)
- Predator care estimează intercept-ul (dacă prada e predictibilă, e mai ușor de prins)

Dacă jinking-ul crește fitness-ul (mai puțin capturat), R-STDP îl întărește.

---

## 8. Decodarea acțiunii — Winner-Take-All

```python
window = spikes_out[-20:]          # ultimii 20 din 100 timesteps
counts = window.sum(axis=0)        # (32,) spike count per neuron output
action = argmax(counts)            # cu tie-breaking aleator
```

De ce ultimii 20? LIF are tranziente inițiale când pornește de la zero. Ultimii 20% din timestep-uri reprezintă răspunsul stabil al rețelei.

**Rețea silențioasă** (zero spikes pe toată fereastra): acțiune aleatoare — exploration implicit, fără mecanism epsilon-greedy separat.

---

## 9. Structura modulelor

```
NeuroGame/
├── game/
│   ├── arena.py          — Dreptunghi, ray-AABB wall distances, coliziune
│   ├── agents.py         — Prey (stamina, arc 180°) + Predator (intercept, 360°)
│   └── engine.py         — Turn loop secvențial, velocity estimation, rewards
├── neuromorphic/
│   ├── encoding.py       — state dict → rates (65,)
│   ├── network.py        — LIF NumPy 65→256→32 + recurrent sparse
│   ├── rstdp.py          — RSTDPTracer + RSTDPController
│   └── decoder.py        — WTA last 20 timesteps
└── experiments/
    └── run.py            — Training loop, CSV logging, weight saving
```

### Separarea responsabilităților

`engine.py` nu importă nimic din `neuromorphic/` — primește `PolicyFn = Callable[[dict], int]`. Asta permite:
- Test mecanică joc cu politici random (fără overhead neuromorphic)
- Înlocuirea backend-ului (NumPy → LAVA → hardware Loihi2) fără modificări în `game/`
- Debugging separat al fiecărui strat

---

## 10. Rulare antrenament

```bash
cd NeuroGame

# Default: 500 episoade, arena 50×50
python -m experiments.run

# Custom
python -m experiments.run --episodes 2000 --seed 7 --arena-w 80 --arena-h 60
```

**Output** în `experiments/logs/`:
- `run_<timestamp>.csv` — statistici per episod (turns, captured, rewards, win rate)
- `weights_<timestamp>.npz` — greutăți finale (W_in, W_rec, W_out pentru ambii agenți)

---

## 11. Roadmap

### Faza 2 — Teren cu elevație
- Izolinii de altitudine (cost locomotor per gradientul pantei)
- Ravine = bariere traversabile cu penalizare stamina crescută
- Reintegrare Unity ca vizualizator (suspendat în faza 1)

### Faza 3 — Hardware Loihi2
- Înlocuire `LIFNumpy` cu procese LAVA native (`lava.proc.lif.models`)
- `encoding.py` → `SpikeGenerator` LAVA
- R-STDP → `LearningRule` LAVA (on-chip learning)

### Faza 4 — Multi-agent
- Populații (N prăzi, M prădători)
- Co-evoluție cu selecție naturală (cei mai slabi agenți reînițializați)
- Analiza emergentei comportamentelor de grup (haită, turmă)
