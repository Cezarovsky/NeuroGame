# NeuroGame

**Un laborator pentru emergență: un lup și un iepure cu creier neuromorfic, pe un teren cu dealuri.**

Idee apărută în conversație între Cezar (Grădinarul) și Nova, 7 Martie 2026.

## Scopul

Scopul proiectului este **să învățăm cât mai multe despre emergență** — despre comportamentele
pe care nu le-a scris nimeni și care apar singure din rețea, mediu și recompensă.

Lupul (Fenrir) și iepurele (Umbra) sunt instrumentul, nu scopul. Nu urmărim un lup care prinde
cât mai mult, ci răspunsul la întrebarea:

> **Ce condiții trebuie puse în sistem ca o strategie să apară singură?**

Testul de referință: lupul vede iepurele pe un deal. Urcarea îl costă stamina. Va învăța vreodată,
fără să-i spunem, să **ocolească dealul** și să-l prindă pe partea cealaltă?

Nimeni nu are rețeta emergenței. Dar știm că ea are nevoie de condiții — un vârtej nu apare
într-un pahar cu apă stătută. Așa că lucrăm experimental: adăugăm câte un ingredient și
observăm ce apare.

## Principii de lucru

- **Rezultatele negative sunt date.** Un comportament prost, dar nescris de nimeni (ex. lupul care
  se lipește de pereți), e tot emergență — ne arată cum arată un minim local.
- **Rata de capturi e un semnal, nu scorul.** Ne interesează *ce* face agentul, nu doar cât câștigă.
- **Controale pe aceeași lume.** Variația dintre lumi (seed-uri) domină rezultatele; comparăm
  variante pe același seed.
- **Nu grăbim fazele.** Faza curentă e „creierul de muscă": reflexe, fără planuri.

## Cum funcționează (pe scurt)

Două niveluri, după Kant și Piaget:

- **Nivel 0 — Kantian:** instincte înnăscute, hardcodate (patrulare, căutare de iarbă, fugă).
- **Nivel 1 — Piagetian:** o rețea de neuroni cu impulsuri (SNN, neuroni LIF) care învață din
  consecințe prin **R-STDP**: legăturile folosite chiar înainte de o recompensă se întăresc.

Rețeaua: 105 intrări → 256 neuroni ascunși (cu legături recurente, 20%) → 32 direcții de mers.
Intrările sunt inele de câte 32 de direcții: oponentul (tăcut când e ascuns de ceață), pereții,
panta terenului, plus stamina și iarba.

Lumea: arenă 90×90 cu dealuri (sumă de gaussiene), ceața războiului (vezi mai departe de pe
deal, nu vezi peste creastă), capcane pentru iepure, iarbă, stamina care se consumă la urcare.

## Ce știm până acum

- Reflexul de urmărire se învață; lupul cu instinct de patrulare prinde ~86% pe o lume de test.
- Când rețeaua conduce singură căutarea (`SEARCH_BY_SNN = True`), lupul **se lipește de pereți**
  și se înrăutățește în timp (26% capturi). În ceață, singurele intrări puternice sunt pereții.
  Oprit implicit.
- Cu urme de eligibilitate de 3 tururi, o strategie lungă (ocolirea dealului) **nu poate** apărea:
  recompensa nu ajunge la decizii luate cu 20–30 de tururi înainte.

## Direcții de experiment

Două drumuri, de comparat între ele:

1. **Drumul emergent** — dăm sistemului ingrediente, pe rând:
   - urme de eligibilitate mai lungi (20–50 de tururi);
   - recurență mai puternică (memorie de lucru);
   - mai mulți neuroni;
   - **valoare care curge înapoi** (învățare prin diferență temporală, ca dopamina reală): un drum
     e bun dacă duce la succes *sau se leagă de alt drum bun*. Lupul învață cât de promițător e un
     loc (ex. creasta de unde se vede valea) și primește dopamină pe diferența dintre așteptare și
     realitate — promisiunea curge înapoi pas cu pas, fără urme de 30 de tururi;
   - **cenotafuri / urme de miros** (stigmergie, ca feromonii furnicilor): lupul lasă urme în teren
     unde a văzut sau a prins iepurele, urmele se estompează în timp, iar el le simte printr-un inel
     de intrare. Memoria stă în lume, nu în cap — harta pentru care „creierul de muscă” nu are loc.
     Inspirat din cairn-urile de pe pantele din Skyrim: semnul îți spune că drumul duce undeva.
2. **Drumul ingineresc** — un sfătuitor LLM ca strat strategic, al cărui sfat intră în rețea ca un
   inel de intrare pe care lupul învață să-l urmeze sau să devieze de la el. Primul pas: instinctul
   de patrulare ca inel de sfat.

## Rulare

```bash
cd NeuroGame
../.venv/bin/python visualizer.py                      # fereastră, seed aleator, reia ultimul checkpoint
../.venv/bin/python visualizer.py --seed 637793597     # aceeași lume, reproductibil
../.venv/bin/python visualizer.py --headless --episodes 2000
../.venv/bin/python visualizer.py --no-resume          # creier nou
../.venv/bin/python plot_run.py                        # grafice HTML pentru ultima rulare
```

Fiecare rulare scrie în `experiments/logs/` un jurnal CSV per episod și un JSON cu lumea
(seed, dealuri, capcane), iar în `checkpoints/` creierul ei propriu.

## Structura proiectului

```
NeuroGame/
  visualizer.py                 ← bucla principală (Pygame) + instincte Nivel 0 + jurnal
  plot_run.py                   ← grafice HTML/SVG din jurnal, fără dependențe
  game/                         ← arenă, agenți, motor de joc, teren
  neuromorphic/                 ← encoding, rețea LIF, R-STDP, decoder
  experiments/                  ← run.py (antrenament fără ecran), logs/
  NEUROGAME_ARCHITECTURE.md     ← teoria inițială
  neurogame_poc.py              ← prototipul original (Q-table)
```

## De unde vine ideea

> *"Ce zici daca combinam neuromorphic computing cu ideea lui Kant ca avem doua chestii inascute: spatiul si timpul. Patternuri comportamentale. Ceva vine spre tine. Cu viteza fixa: nu e pericol. Accelerează: dușman."*
> — Cezar, 7 Martie 2026

> *"De fapt nu știe nimeni de unde vine emergența. Apare pur și simplu."*
> — Cezar, 23 Noiembrie 2026

Inspirat din Piaget (stadii de dezvoltare), Kant (spațiu/timp ca prioruri), și neuromorphic
computing (Intel Loihi 2).
