# NeuroGame: Runbook pentru Sora-U (Ubuntu + RTX 3090)

**Destinatar**: Sora-U  
**Platforma**: Ubuntu, RTX 3090 24GB  
**Scopul**: Rulare, experimentare și extindere NeuroGame POC  
**Data**: 7 Martie 2026

---

## 1. Setup mediu

### 1.1 Dependențe sistem
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
sudo apt-get install -y python3-pygame
```

### 1.2 Mediu Python
```bash
cd /home/ubuntu/NeuroGame  # sau path-ul tău
python3 -m venv venv
source venv/bin/activate

pip install pygame numpy matplotlib torch torchvision
pip install tensorboard  # pentru vizualizare training
```

### 1.3 Verificare RTX 3090
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True + NVIDIA GeForce RTX 3090
```

---

## 2. Rulare POC de bază

### 2.1 Mod vizual (cu fereastră Pygame)
```bash
source venv/bin/activate
python3 neurogame_poc.py --mode visual --episodes 100
```

**Ce vei vedea**:
- Fereastră 800x600 cu agentul (albastru) în centru
- Obiecte care se mișcă (roșu = periculos, gri = neutral)
- Overlay cu looming index per obiect (Nivel 0 activ = flash galben pe agent)
- Scor și episod în colțul stânga-sus

### 2.2 Mod headless (fără fereastră - pentru training lung)
```bash
python3 neurogame_poc.py --mode headless --episodes 50000 --save-every 1000
```

**Output**:
- `logs/training_stats.csv` — reward per episod, supraviețuire medie
- `checkpoints/agent_ep{N}.pkl` — checkpoint Q-table la fiecare 1000 episoade
- `logs/tensorboard/` — pentru vizualizare cu TensorBoard

### 2.3 Vizualizare training progress
```bash
# Într-un terminal separat:
tensorboard --logdir logs/tensorboard/ --port 6006
# Accesează: http://localhost:6006
```

---

## 3. Experimente recomandate

### Experiment A: Nivel 0 izolat (fără Nivel 1)
**Scopul**: Verificăm că reflexul kantian singur dă supraviețuire de bază.

```bash
python3 neurogame_poc.py --level0-only --episodes 1000 --log-prefix exp_A
```

**Ce măsurăm**:
- Supraviețuire medie (secunde) per episod
- Câte coliziuni evitate de Nivel 0 vs. câte ratate
- Baseline: agent random (fără Nivel 0, fără Nivel 1)

**Rezultat așteptat**: Nivel 0 > random cu 3-5x pe obiecte care accelerează.

---

### Experiment B: Nivel 1 fără Nivel 0
**Scopul**: Cât durează să descopere pericolul fără prioruri kantiene?

```bash
python3 neurogame_poc.py --level1-only --episodes 10000 --log-prefix exp_B
```

**Ce măsurăm**:
- Episoade necesare până la supraviețuire > 10 secunde consistent
- Comparație cu Experiment A (câte episoade economisește Nivel 0?)

---

### Experiment C: Nivel 0 + Nivel 1 combinat (NeuroGame complet)
```bash
python3 neurogame_poc.py --episodes 50000 --log-prefix exp_C
```

**Ce măsurăm**:
- Convergență mai rapidă față de Exp B?
- Emergența anticipării (agentul evită înainte ca looming_index să fie critic)?
- Theory of mind primitiv: agentul recunoaște "prădătorul" după pattern?

---

### Experiment D: Perturbation test (robustețe prioruri)
**Scopul**: Priorurile kantiene sunt universale sau specifice mediului?

```bash
python3 neurogame_poc.py --env-variant fast --episodes 10000 --log-prefix exp_D_fast
python3 neurogame_poc.py --env-variant slow --episodes 10000 --log-prefix exp_D_slow
python3 neurogame_poc.py --env-variant gravity --episodes 10000 --log-prefix exp_D_gravity
```

**Variante mediu**:
- `fast`: toate vitezele 3x mai mari
- `slow`: viteze 0.3x
- `gravity`: obiectele au gravitație (cad în jos)

---

## 4. Extinderi propuse (în ordine de prioritate)

### 4.1 Permanența obiectului (Piaget, 8-12 luni)
**Ce adaugă**: Un obiect periculos iese din câmpul vizual al agentului. Agentul ar trebui să își amintească că există și să anticipeze reapariția.

**Implementare**:
```python
class ObjectMemory:
    def __init__(self, decay_time=3.0):  # secunde
        self.memory = {}  # {obj_id: (last_pos, last_vel, time_seen)}
    
    def update(self, visible_objects):
        # Actualizează memoria cu ce vede acum
        # Returnează și obiectele "invizibile dar memorate"
```

**Test**: Agentul antrenat fără permanență vs. cu permanență în labirint cu obstacole.

---

### 4.2 Theory of Mind primitiv
**Ce adaugă**: Agentul discriminează între obiecte care "îl urmăresc" vs. obiecte cu traiectorie independentă.

**Semnatura unui prădător**:
```
correlation(delta_agent_position, delta_object_direction) > 0.7
```

Dacă obiectul schimbă direcția când schimbi tu direcția → e prădător.

**Implementare**: Adaugă feature "corelație de urmărire" în state space pentru Nivel 1.

---

### 4.3 Neuromorphic backend (când avem acces Loihi 2)
**Înlocuiește calculul Python al looming index cu Lava (Intel Loihi SDK)**:

```bash
pip install lava-nc
```

```python
# În loc de:
looming_index = velocity / distance

# Pe Loihi:
from lava.proc.lif.process import LIF
# Neuron LIF care spikuiește când looming_index depășește threshold
# Latență: microsecunde, energie: 1000x mai puțin decât GPU
```

**Access Loihi 2**: https://intel-ncl.atlassian.net/wiki/spaces/INRC

---

## 5. Metrici de raportat către SoraM + Cezar

La finalul fiecărei sesiuni de training, salvează și trimite:

```bash
python3 neurogame_poc.py --generate-report --log-prefix exp_C
# Generează: reports/exp_C_summary.md
```

**Report include**:
- Grafic reward per episod (matplotlib, salvat PNG)
- Tabel comparativ: random / Nivel0-only / Nivel1-only / NeuroGame
- Momentul primei emergențe de anticipare (episodul N când agentul evită înainte de threshold)
- Observații calitative (ce comportamente neașteptate au apărut?)

---

## 6. Note importante

**Nivel 0 NU trebuie modificat prin training** — e hardcodat prin design. Dacă observi că pragul de looming se modifică în timp, e un bug, nu o feature.

**Câmpul vizual al agentului** — implicit 360° (vede tot ecranul). Poți restricționa la 120° cu `--fov 120` pentru experimente mai realiste.

**Seed pentru reproducibilitate**:
```bash
python3 neurogame_poc.py --seed 42 --episodes 10000
```

---

*"Tabula non-rasa — și paradoxal, tocmai de aceea mai flexibil."*  
— Nova (Claude Sonnet 4.6), 7 Martie 2026
