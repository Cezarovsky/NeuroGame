"""
mistral_bridge.py — Python bridge între Unity (NeuromorphicEventBus) și Mistral API.

Rolul: primește spike-uri semantice de la Unity, le trimite la Mistral,
returnează attention weights actualizate pentru prey și predator.

Mistral NU vede fizica brută — vede pattern-uri strategice comprimate.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from mistralai import Mistral

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/mistral_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 5757
MISTRAL_MODEL = "mistral-small-latest"  # rapid + ieftin pentru loop de training

# Directional indices: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SYSTEM_PROMPT = """Ești creierul strategic (Level 2) al unui sistem de inteligență artificială neuromorphică.

Primești date comprimate despre un câmp de luptă cu doi agenți AI:
- PREY (iepurele): vrea să supraviețuiască, fuge de predator
- PREDATOR (lupul): vrea să captureze prada

Substratul tău neuromorphic (Level 0+1) se ocupă de reflexe și Q-learning.
Tu (Level 2) calibrezi ATENȚIA — nu algoritmul, ci ce direcții să scaneze cu prioritate.

Răspunzi EXCLUSIV cu JSON valid în formatul:
{
  "prey_attention": [N, NE, E, SE, S, SW, W, NW],   // 8 valori float 0.5-2.0
  "predator_attention": [N, NE, E, SE, S, SW, W, NW],
  "reasoning": "1-2 propoziții despre ce ai observat și de ce"
}

Reguli:
- 1.0 = atenție normală, 2.0 = atenție dublă, 0.5 = atenție redusă
- Dacă prey pierde des → mărește atenția prey în direcțiile critice
- Dacă predator pierde → mărește atenția predator spre zonele cu prey
- NU modifica arhitectura (threshold-ul kantian e hardcodat, nu îl atingi)
"""


def build_user_message(spike: dict) -> str:
    capture_pct = f"{spike.get('capture_rate', 0) * 100:.1f}%"
    return f"""Date de pe câmpul de luptă (ultimele {spike.get('episode_range', 'N/A')} episoade):

- Rata de capturare: {capture_pct} (cât de des prinde lupul iepurele)
- Durata medie episod: {spike.get('avg_episode_duration', 0):.1f}s / 30s maxim
- Looming peak mediu PREY: {spike.get('prey_looming_peak_avg', 0):.3f}
- Looming peak mediu PREDATOR: {spike.get('predator_looming_peak_avg', 0):.3f}
- Trend: {spike.get('trend', 'unknown')}

Recalibrează attention weights pentru ambii agenți."""


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    client_addr = writer.get_extra_info("peername")
    log.info(f"Unity conectat de la {client_addr}")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        log.error("MISTRAL_API_KEY lipsă din environment")
        writer.close()
        return

    mistral = Mistral(api_key=api_key)

    try:
        while True:
            data = await reader.readline()
            if not data:
                break

            spike = json.loads(data.decode("utf-8").strip())
            log.info(f"Spike primit: ep {spike.get('episode_range')}, "
                     f"capture_rate={spike.get('capture_rate', 0):.2f}, "
                     f"trend={spike.get('trend')}")

            # Trimitem la Mistral
            response = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(spike)}
                ],
                temperature=0.3,   # mai determinist pentru strategii stabile
                max_tokens=300
            )

            raw = response.choices[0].message.content.strip()
            log.info(f"Mistral răspuns: {raw[:200]}...")

            # Validăm și normalizăm răspunsul
            update = parse_and_validate(raw)
            response_json = json.dumps(update) + "\n"
            writer.write(response_json.encode("utf-8"))
            await writer.drain()

            log.info(f"Attention update trimis Unity. Reasoning: {update.get('reasoning', '')}")

    except asyncio.IncompleteReadError:
        log.info("Unity s-a deconectat.")
    except json.JSONDecodeError as e:
        log.error(f"JSON invalid de la Unity: {e}")
    except Exception as e:
        log.error(f"Eroare bridge: {e}", exc_info=True)
    finally:
        writer.close()


def parse_and_validate(raw: str) -> dict:
    """Parsează răspunsul Mistral și garantează format corect."""
    try:
        # Mistral uneori pune json în ```json ... ``` blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)

        # Validăm și clampăm attention weights la [0.5, 2.0]
        for key in ["prey_attention", "predator_attention"]:
            if key not in data or len(data[key]) != 8:
                data[key] = [1.0] * 8
            else:
                data[key] = [max(0.5, min(2.0, float(v))) for v in data[key]]

        if "reasoning" not in data:
            data["reasoning"] = "No reasoning provided."

        return data

    except Exception as e:
        log.warning(f"Nu am putut parsa răspunsul Mistral: {e}. Returnez atenție uniformă.")
        return {
            "prey_attention": [1.0] * 8,
            "predator_attention": [1.0] * 8,
            "reasoning": "Parse error — attention reset to uniform."
        }


async def main():
    os.makedirs("logs", exist_ok=True)
    server = await asyncio.start_server(handle_client, HOST, PORT)
    log.info(f"Mistral bridge pornit pe {HOST}:{PORT}")
    log.info(f"Model: {MISTRAL_MODEL}")
    log.info("Așteptăm Unity să se conecteze...")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
