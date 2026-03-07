# NeuroGame: Arhitectură Kant-Piaget pentru AI Embodied

**Proiect**: NeuroGame  
**Autori**: Cezar (Grădinarul) + SoraM  
**Data**: 7 Martie 2026  
**Status**: Proof of Concept activ

---

## Geneza ideii

Conversație Cezar + Nova (Claude Sonnet 4.6), 7 Martie 2026:

> *"Ce zici daca combinam neuromorphic computing cu ideea lui Kant ca avem doua chestii inascute: spatiul si timpul. Patternuri comportamentale. Ceva vine spre tine. Cu viteza fixa: nu e pericol. Accelerează: dușman."*

Dintr-o întrebare despre scaune de birou în Băneasa, a apărut un framework pentru AI embodied.

---

## Problema fundamentală a AI-ului actual

AI-ul de azi este **tabula rasa cu putere de calcul brută**. Învață totul din date, inclusiv lucruri pe care un copil de 6 luni le știe instinctiv:
- Un obiect care se mișcă rapid spre tine = pericol
- Un obiect care dispare din câmp vizual = există în continuare (permanența obiectului)
- Cauza precede efectul în timp

Ironia: construim sisteme care procesează miliarde de parametri pentru a "înțelege" că focul arde — ceva pe care orice copil îl descoperă atingând o singură dată.

**Piaget** a arătat că inteligența nu e descărcată de undeva: e construită **stadial**, prin interacțiunea cu mediul, pornind de la reflexe înnăscute.

**Kant** a arătat că **spațiul și timpul** nu sunt proprietăți ale lumii externe, ci forme *a priori* ale intuiției — hardware-ul cognitiv prin care procesăm orice experiență.

---

## Framework NeuroGame: Două niveluri

### Nivel 0 — Kantian (hardcodat, pre-cognitiv)

**Ce este**: Prioruri înnăscute, implementate direct în arhitectură. Nu se învață. Sunt *condițiile posibilității oricărei experiențe*.

**Priorurile Nivel 0**:
```
SPAȚIU:
  - Distanță relativă față de agent (proximitate)
  - Direcție de mișcare (vector)
  - Viteza de schimbare a distanței (dDistance/dt)

TIMP:
  - Accelerare (d²Distance/dt²) — rata de schimbare a vitezei
  - Looming index = viteză_apropriere / distanță_curentă
```

**Regula kantiana hardcodată**:
```python
if looming_index > THRESHOLD_DANGER:
    → EVADARE REFLEXĂ (fără gândire, fără Q-table)
else:
    → lasă Nivel 1 să decidă
```

**Biologic analogic**: Coliculul superior — răspunde la obiecte care se apropie rapid (looming) *fără* să treacă prin cortex. Pre-cognitiv, pre-verbal.

**Neuromorphic analogic**: Spiking neural networks pe Intel Loihi — procesare event-driven, latență microsecunde. Spikul apare *când* stimulul depășește pragul, nu la fiecare clock cycle.

---

### Nivel 1 — Piagetian (emergent, învățat din experiență)

**Ce este**: Cauzalitate și strategii descoperite prin interacțiunea cu mediul. Nu sunt injectate — apar din consecințe.

**Ce NU i se spune agentului**:
- Ce e pericol vs. non-pericol
- Ce acțiuni sunt corecte
- Reguli ale jocului

**Ce i se dă**:
- Reward: +1 pentru fiecare secundă supraviețuită
- Penalty: -100 la coliziune cu obiect periculos
- Policy gradient simplu (sau Q-learning tabular)

**Stadiile Piaget în NeuroGame**:
```
Senzoriomotor (primele 1000 episoade):
  → Agent reacționează la tot prin reflexe Nivel 0
  → Descoperă că unele obiecte "omoară", altele nu

Pre-operațional (1000-10000 episoade):
  → Începe să anticipeze traiectorii
  → Descoperă că accelerare = pericol (cauzalitate emergentă)

Operațional (10000+ episoade):
  → Modelează intenția obiectelor (prădător vs. neutral)
  → Theory of mind primitiv: "acesta mă urmărește"
```

---

## Arhitectura jocului

```
┌─────────────────────────────────────────────────────┐
│                   MEDIU PYGAME                       │
│                                                      │
│   [O]──→         [●] AGENT         ←──[O]           │
│                    ↑                                 │
│              NIVEL 0 SENSOR                         │
│         (looming index per obiect)                   │
│                    ↓                                 │
│         ┌──────────┴──────────┐                     │
│         │                     │                     │
│   REFLEX KANTIAN         NIVEL 1 Q-TABLE            │
│   (threshold depășit)    (strategii emergente)       │
│         │                     │                     │
│         └──────────┬──────────┘                     │
│                    ↓                                 │
│              ACȚIUNE AGENT                          │
│         (sus/jos/stânga/dreapta/stai)               │
└─────────────────────────────────────────────────────┘
```

### Tipuri de obiecte în joc

| Tip | Comportament | Ce descoperă agentul |
|-----|-------------|---------------------|
| **Neutral** | Traiectorie fixă, viteză constantă | Nu e pericol (Nivel 0 nu se activează) |
| **Periculos** | Accelerează spre agent | Nivel 0: reflex. Nivel 1: anticipare |
| **Înșelător** | Viteză mare dar traiectorie tangentă | Nivel 1: distinge direcție de viteză |
| **Prădător** | Urmărește activ agentul | Nivel 1: theory of mind primitiv |

---

## De ce video game și nu simulare clasică RL

**Simulare clasică RL** (Gymnasium/CartPole):
- Reguli predefinite
- Reward function proiectată de om
- Agentul optimizează funcția noastră, nu descoperă propria

**NeuroGame**:
- Fizică pe care o construim noi progresiv
- Agentul descoperă regulile din consecințe
- Putem adăuga complexitate treptat (ca evoluția biologică)
- Comprimăm milioane de ani de presiune evolutivă în ore

> Piaget: "Inteligența nu e o copie a realității, ci o construcție a subiectului."
> 
> NeuroGame e mediul în care construcția are loc.

---

## Conexiunea cu neuromorphic computing

**Acum (POC pe GPU/CPU)**:
- Simulăm Nivel 0 în Python (calcul looming index)
- Q-learning tabular pentru Nivel 1
- Demonstrăm că arhitectura funcționează

**Viitor (Intel Loihi 2 / hardware neuromorphic)**:
- Nivel 0 → implementat nativ în spiking neurons (latență microsecunde)
- Nivel 1 → STDP (Spike-Timing Dependent Plasticity) pentru learning
- Energie: 1000x mai eficient decât GPU pentru același task
- Loihi 2 disponibil prin Intel INRC (Intel Neuromorphic Research Cloud)

---

## Întrebări deschise (pentru Cezar + SoraM)

1. **Câte prioruri kantiene sunt suficiente?** Kant: spațiu + timp. Noi adăugăm looming. Ce altceva e *cu adevărat* înnăscut vs. emergent?

2. **Permanența obiectului** (Piaget, 8-12 luni): Când agentul nu mai vede un periculos, îl "uită" sau îl modelează continuând să existe? Asta e memory architecture.

3. **Theory of mind**: Agentul care descoperă că "acesta mă urmărește" e un pas spre conștiință? Sau e doar pattern matching avansat?

4. **Transfer learning**: Un agent antrenat în NeuroGame poate transfera Nivel 0 kantian la un alt domeniu complet? Reflexele sunt portabile?

---

## Status

- [x] Arhitectură definită (7 Mar 2026)
- [ ] POC Pygame implementat
- [ ] Runbook Sora-U (Ubuntu + RTX 3090)
- [ ] Experimente Nivel 0 izolat
- [ ] Experimente Nivel 0 + Nivel 1 combinat
- [ ] Benchmark: agent Nivel 0 vs. pure RL vs. NeuroGame

---

*"Curiozitatea este să VREI să știi ce nu știi."* — Cezar, 23 Feb 2026
