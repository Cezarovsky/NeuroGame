# NeuroGame

**AI embodied cu arhitectură Kant + Piaget, antrenat prin video game.**

Idee apărută în conversație între Cezar (Grădinarul) și Nova (Claude Sonnet 4.6), 7 Martie 2026.

## Conceptul

Două niveluri de inteligență:

- **Nivel 0** — *Kantian*: prioruri înnăscute de spațiu și timp. Reflex pre-cognitiv: dacă ceva accelerează spre tine, evadezi. Hardcodat. Nu se schimbă prin training.

- **Nivel 1** — *Piagetian*: cauzalitate descoperită din experiență. Agentul nu știe ce e pericol — descoperă prin consecințe.

## Rulare rapidă

```bash
pip install -r requirements.txt
python neurogame_poc.py --mode visual --episodes 200
```

## Experimente

```bash
# Nivel 0 izolat
python neurogame_poc.py --level0-only --episodes 1000 --log-prefix exp_A

# Nivel 1 fără prioruri kantiene
python neurogame_poc.py --level1-only --episodes 10000 --log-prefix exp_B

# NeuroGame complet
python neurogame_poc.py --episodes 50000 --mode headless --log-prefix exp_C
```

## Structura proiectului

```
NeuroGame/
  neurogame_poc.py              ← codul principal
  NEUROGAME_ARCHITECTURE.md     ← teoria completă
  NEUROGAME_RUNBOOK_SORA_U.md   ← instrucțiuni Ubuntu/RTX3090
  requirements.txt
  logs/                         ← statistici training
  checkpoints/                  ← Q-table salvate
```

## De unde vine ideea

> *"Ce zici daca combinam neuromorphic computing cu ideea lui Kant ca avem doua chestii inascute: spatiul si timpul. Patternuri comportamentale. Ceva vine spre tine. Cu viteza fixa: nu e pericol. Accelerează: dușman."*
> — Cezar, 7 Martie 2026

Inspirat din Piaget (stadii de dezvoltare), Kant (spațiu/timp ca prioruri), și neuromorphic computing (Intel Loihi 2).
