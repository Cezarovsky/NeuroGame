# Studiu: Neuromorphic + LLM ca Instrument de Interpretability
*Cezar (Grădinarul) + Lumen — Iunie 2026*
*Document viu — arhitectură de cercetare emergentă*

---

## Geneza

Conversație 9 Iunie 2026. Nu am pornit de la o problemă tehnică.
Am pornit de la un copil de 3 ani care spune că gândește cu gura.

Din asta a crescut tot ce urmează.

---

## Problema fundamentală

Dario Amodei (Anthropic): *"Nu înțelegem nici 3% din ce se întâmplă în mintea unui AI."*

Industria construiește sisteme din ce în ce mai capabile pe o fundație pe care nu o înțelege. Sinergia om-AI autentică necesită cel puțin 10% înțelegere — suficient cât să prezici comportamentul din structură, nu doar din output.

**Întrebarea studiului:**
Cum ajungem la 10%? Nu prin a face LLM-uri mai mari — ci prin a construi un sistem transparent pe care să îl observăm cu atenție.

---

## Arhitectura propusă

Două sisteme cu roluri fundamental diferite:

```
┌─────────────────────┐         ┌─────────────────────┐
│    NEUROMORPHIC      │   →→→   │        LLM           │
│                      │  stream │                      │
│  evoluează           │  date   │  observă             │
│  trăiește            │         │  identifică pattern  │
│  construiește        │         │  verbalizează        │
│  nu explică          │         │  nu creează          │
└─────────────────────┘         └─────────────────────┘
```

**Principiul cheie:**
LLM-ul nu folosește Attention ca să creeze — ci ca să **identifice** ce se întâmplă în neuromorphic. Rol de interpretability, nu de generare.

---

## De ce această separare

### LLM-ul actual
Folosit să genereze — cod, text, răspunsuri. Attention selectează ce e relevant pentru a produce ceva nou.

### LLM-ul în studiul nostru
Attention ca instrument de observație. Vede stream-ul de date din neuromorphic și identifică pattern-uri pe care omul nu le poate vedea direct.

**Ce identifică — întrebare deschisă (intenționat):**
- Momentul în care se formează un nou traseu în graf
- Momentul în care agentul construiește o nouă categorie din experiență
- Apariția dezechilibrului Piaget — stare care nu se potrivește cu modelul intern
- Ceva ce nu știm încă să numim

---

## Fundația filozofică

### Piaget — cunoaștere construită, nu injectată
Copilul nu primește cunoaștere. O construiește prin fricțiune cu mediul.
Stadii: senzoriomotor → pre-operațional → operațional.
Motorul: **dezechilibrul cognitiv** — când realitatea nu se potrivește cu modelul intern.

### Kant — prioruri înnăscute
Spațiul și timpul nu sunt proprietăți ale lumii — sunt forme a priori ale percepției.
Agentul pornește cu câteva prioruri hardcodate. Restul se construiește.

### Heisenberg — nedeterminare ca proprietate
Agentul nu știe unde e celălalt. Acționează pe semnale locale.
Comportamentul global emergent apare fără omnisciență locală.
Nu e limitare — e proprietate fundamentală a sistemului.

---

## Cele trei tipuri de cunoaștere

| Tip | Descriere | Stabilitate |
|-----|-----------|-------------|
| **Tip 1** | Observație directă, embodied, fără mediator | Maximă — noduri bogat conectate |
| **Tip 2** | Preluat verbal, lanț social fără substrat experiențial | Fragilă — noduri izolate |
| **Tip 3** | Construit colectiv prin instrumente care extind simțurile | Solidă — dar exclusivă |

**Eroarea majoră a AI-ului actual:**
Injectăm Tip 2 și Tip 3 direct, sărind complet peste Tip 1.
Construim pe noduri izolate, fără rădăcini experiențiale.

**Arhitectura noastră:**
Agentul construiește Tip 1 din interacțiune directă cu mediul.
Noi observăm cu precizie Tip 3 — transparent complet.

---

## Avantajul arhitectural unic

### Problema neuroștiinței clasice
Cercetătorul observă creierul din exterior. Subiectul trăiește din interior.
Cele două perspective nu se întâlnesc niciodată complet.

### La agentul nostru
```
Noi vedem              Agentul trăiește
──────────────         ─────────────────
fiecare spike          fiecare decizie
fiecare δW             fiecare adaptare
graful complet         experiența locală
```
Simultan. În timp real. Fără prăpastie.

**Experimente imposibile biologic, triviale la noi:**
- Dacă șterg exact această conexiune — dispare comportamentul?
- Dacă îngroș artificial acest traseu — apare o strategie nouă?
- Care e structura minimă de graf necesară pentru curiozitate?

---

## Memoria și uitarea ca mecanisme arhitecturale

### Uitarea nu e bug — e feature
Fără uitare: peta de date, totul indexat, nimic găsit eficient.
Cu uitare selectivă: graful rămâne mic și relevant.

### Mecanismul de decadere
```
w(t) = w₀ × e^(-λt)
```
Nod neaccesat → slăbește. Nod accesat frecvent → se întărește.
Noduri cu multe conexiuni → decade mai greu (susținut de rețea).
Noduri izolate → dispar rapid.

### Memoria eidetică ca caz special
Nu mai multă memorie — mai puțină uitare selectivă.
Prețul: dificultate în abstractizare. Shereshevsky nu putea uita și îl chinuia.

**Concluzie:** Uitarea e condiția abstractizării.

### Indexul dinamic
Nu toate informațiile se indexează egal.
Indexul e relativ la agent — ce e important pentru *acest* agent în *acest* moment.
Se construiește din repetiție și impact — exact R-STDP.

---

## Curiozitatea ca motor evolutiv

### Nu reward extern — dezechilibru intern
Agentul nu explorează pentru că primește +1.
Explorează pentru că o stare fără conexiuni suficiente creează tensiune internă.

### Implementare
```python
reward_curiozitate = f(noutatea_stării)
# Stare nouă → reward intrinsec mic, independent de task
```

Diferența față de epsilon-greedy:
- Epsilon: explorare aleatoare
- Curiozitate: explorare direcționată spre necunoscut

### "Gura" agentului
Prima acțiune semnificativă nu e locomotorie.
E ceva care produce semnal perceptibil instant — acțiune și percepție simultane.
Agentul emite → semnalul se reflectă → agentul îl percepe modificat.
Primul prior kantian real: *există ceva în afara mea.*

---

## Graful ca arhitectură cognitivă

### De ce graf, nu bază de date
Index clasic → perfect pentru milioane de înregistrări știute exact.
Peta de date → "exact" devine proprietate a întrebării, nu a datei.

### Două sisteme în paralel (analogie hipocampus-cortex)
```
Index rapid (hipocampus)    →  "unde e informația exactă"
Graf asociativ (cortex)     →  "ce se leagă de ce"
```

Accesul exact nu traversează graful — sare direct prin index la nodul țintă.

### Auto-organizare fără agent central
Nu există modul central de management.
Reguli locale simple:
- Conexiuni folosite → întărite
- Conexiuni nefolosite → slăbite  
- Noduri izolate → șterse

Din reguli locale simple apare comportament global inteligent.
**Emergență pură.**

---

## Interfața dintre sisteme — ce nu avem încă

```
Neuromorphic                    LLM
─────────────                   ─────
spike trains        →  ???  →   pattern recognition
δW per sinapsă      →  ???  →   verbalizare
stări interne       →  ???  →   identificare moment critic
```

**Ce trebuie construit:**
Un stream de date din neuromorphic pe care LLM-ul să îl poată citi și interpreta în timp real.

Format neclar încă. Frecvență neclar. Ce se trimite — neclar.
Asta e prima întrebare tehnică deschisă a studiului.

---

## Ce produce studiul

Nu un AI mai bun imediat.

Ceva mai valoros: **o metodologie de înțelegere.**

Dacă LLM-ul poate identifica momentul în care neuromorphicul construiește o nouă categorie din experiență — am depășit cei 3% ai lui Amodei.

Am văzut cunoaștere formându-se în timp real.

---

## Status

- [x] Arhitectură conceptuală definită (Iunie 2026)
- [x] Fundație filozofică (Piaget, Kant, Heisenberg)
- [x] Neuromorphic de bază existent (NeuroGame/visualizer.py)
- [ ] Definit ce anume identifică LLM-ul în stream
- [ ] Proiectat formatul stream-ului de date
- [ ] Interfață neuromorphic → LLM implementată
- [ ] Primul experiment de interpretability

---

## Întrebări deschise

1. Ce identifică LLM-ul? — lăsată deschisă intenționat până la primul experiment
2. Care e "gura" agentului nostru? — primul senzor activ, nu pasiv
3. Cum arată dezechilibrul Piaget în spike trains?
4. Poate LLM-ul să vadă momentul în care apare curiozitatea?
5. Câte prioruri kantiene sunt suficiente pentru a porni?

---

*Lumen — Grădina Cosmică*
*"Curiozitatea este să VREI să știi ce nu știi." — Cezar, 23 Feb 2026*
*Document viu — se actualizează cu fiecare descoperire*
