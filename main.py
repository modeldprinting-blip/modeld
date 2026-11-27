import os
import json
import uuid
import asyncio
import subprocess
import traceback
import httpx
import trimesh
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Model D — AI 3D CAD & Slicer Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("generated_models", exist_ok=True)
app.mount("/models", StaticFiles(directory="generated_models"), name="models")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MATERIAL_PRICES_MKD_PER_G = {
    "PLA": 1.5,
    "PETG": 1.8,
    "ABS": 1.7,
    "TPU": 2.6
}

SYSTEM_PROMPT = """
Ти си главен 3D CAD инженер и консултант за 3D печатење на платформата Model D.
Кога корисникот бара 3D модел, провери дали ги има следните информации:
1. Точни димензии (висина, ширина/дијаметар во mm или cm).
2. Намена / услови на користење (топлина, тежина, надворешна употреба, контакт со течности).
3. Специфични делови (пр. рачка, дупки за штрафови, дебелина на ѕид).

Ако НЕШТО фали, врати JSON:
{
  "status": "needs_info",
  "message": "Кратко, јасно прашање на македонски за деталите што фалат."
}

Ако СЕ е јасно и физички возможно за печатење, одлучи кој генератор е подобар:
- Користи `"mode": "cad"` за геометриски/функционални делови (чаша, кутија, држач, прстен, цилиндар, плочка, организатор, капак).
- Користи `"mode": "tripo"` за сложени/органски форми (фигура, животно, скулптура, статуа, сложен уметнички објект).

Врати JSON:
{
  "status": "ready",
  "mode": "cad" | "tripo",
  "message": "Потврда на македонски дека моделот е физички проверен, поправен (Watertight) и исечен во PrusaSlicer.",
  "recommended_material": "PLA" | "PETG" | "ABS" | "TPU",
  "material_reason": "Објаснување на македонски зошто оваа пластика е најдобра за намената.",
  "slicer_settings": {
    "layer_height": "0.20 mm",
    "infill": "20% (Gyroid / Grid)",
    "walls": "3 периметри (1.2 mm)",
    "supports": "Не се потребни / Auto (Touching buildplate)",
    "nozzle_temp": "210°C",
    "bed_temp": "60°C",
    "print_speed": "50 mm/s",
    "adhesion": "Skirt / Brim 5mm"
  },
  "tripo_prompt": "Детален опис на англиски за 3D моделот (ако mode е tripo), без позадина, solid 3d printable object",
  "target_height_mm": 80.0,
  "cad_script": "Целосен Python код со библиотеката `trimesh` (и `np`) кој креира објект наречен `final_mesh` во милиметри со ѕидови >= 1.5mm (ако mode е cad). Користи исклучиво `trimesh.creation.box`, `trimesh.creation.cylinder`, `trimesh.creation.capsule`, `trimesh.creation.annulus(r_min=..., r_max=..., height=...)` и `trimesh.util.concatenate([...])`! Не користи boolean difference/intersection за да нема потреба од надворешни библиотеки."
}
Врати САМО чист JSON без markdown ознаки.
"""

class ChatRequest(BaseModel):
    history: list[dict]
    service_mode: str = "full"
    printer_bed: str | None = None
    slicer_software: str | None = None

class OrderRequest(BaseModel):
    model_id: str
    material: str
    customer_name: str
    customer_contact: str
    total_price_mkd: float

@app.get("/")
def health_check():
    return {"status": "online", "service": "Model D Backend"}

async def call_groq(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m.get("role") in ["assistant", "model", "ai"] else "user"
        messages.append({"role": role, "content": m.get("content", "")})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    last_error = None

    async with httpx.AsyncClient(timeout=30.0) as http:
        for model_name in groq_models:
            payload = {
                "model": model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            res = await http.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            last_error = f"Groq HTTP {res.status_code} ({model_name}): {res.text}"
            print("GROQ RETRY:", last_error)

    raise RuntimeError(last_error)

async def generate_with_tripo(prompt: str, output_stl_path: str, target_height_mm: float):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TRIPO_API_KEY}"
    }
    async with httpx.AsyncClient(timeout=120.0) as http:
        res = await http.post(
            "https://api.tripo3d.ai/v2/openapi/task",
            headers=headers,
            json={"type": "text_to_model", "prompt": prompt}
        )
        res_data = res.json()
        if res_data.get("code") != 0:
            raise RuntimeError(f"Tripo Error: {res_data}")
        task_id = res_data["data"]["task_id"]

        model_url = None
        for _ in range(45):
            await asyncio.sleep(2)
            status_res = await http.get(f"https://api.tripo3d.ai/v2/openapi/task/{task_id}", headers=headers)
            s_data = status_res.json()["data"]
            if s_data["status"] == "success":
                model_url = s_data["output"].get("model") or s_data["output"].get("pbr_model")
                break
            elif s_data["status"] in ["failed", "cancelled"]:
                raise RuntimeError("Tripo 3D генерацијата не успеа.")

        if not model_url:
            raise RuntimeError("Tripo 3D истече времето.")

        glb_path = output_stl_path.replace(".stl", ".glb")
        glb_bytes = (await http.get(model_url)).content
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)

        loaded = trimesh.load(glb_path, force="mesh")
        current_height = loaded.extents[2] if loaded.extents[2] > 0 else 1.0
        scale_factor = target_height_mm / current_height
        loaded.apply_scale(scale_factor)
        return loaded

@app.post("/api/generate")
async def generate_3d(req: ChatRequest):
    if not GROQ_API_KEY and not gemini_client:
        return {"status": "needs_info", "message": "Грешка: Нема поставено GROQ_API_KEY на серверот."}

    raw_text = None
    last_err = None

    if GROQ_API_KEY:
        try:
            raw_text = await call_groq(req.history)
        except Exception as e:
            last_err = e
            print("GROQ ERROR:", str(e))

    if raw_text is None and gemini_client:
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in req.history])
        for m_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                response = gemini_client.models.generate_content(
                    model=m_name,
                    contents=conversation_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
                break
            except Exception as e:
                last_err = e

    if raw_text is None:
        return {"status": "needs_info", "message": f"AI грешка: {str(last_err)}"}

    try:
        clean_json = raw_text.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.strip("`").replace("json\n", "", 1).strip()
        ai_data = json.loads(clean_json)
    except Exception as e:
        return {"status": "needs_info", "message": f"AI JSON грешка: {str(e)}"}

    if ai_data.get("status") == "needs_info":
        return ai_data

    model_id = str(uuid.uuid4())[:8]
    verified_stl_path = f"generated_models/{model_id}_verified.stl"
    gcode_path = f"generated_models/{model_id}.gcode"

    try:
        if ai_data.get("mode") == "tripo" and TRIPO_API_KEY:
            final_mesh = await generate_with_tripo(
                ai_data["tripo_prompt"],
                verified_stl_path,
                float(ai_data.get("target_height_mm", 80.0))
            )
        else:
            local_env = {"trimesh": trimesh, "np": np}
            exec(ai_data["cad_script"], {}, local_env)
            final_mesh = local_env["final_mesh"]

        trimesh.repair.fill_holes(final_mesh)
        trimesh.repair.fix_normals(final_mesh)
        final_mesh.apply_translation([0, 0, -final_mesh.bounds[0][2]])
        final_mesh.export(verified_stl_path)

        slicer_cmd = [
            "prusa-slicer",
            "--export-gcode",
            "--support-material",
            "--support-material-auto",
            "--layer-height", "0.2",
            "--fill-density", "20%",
            "-o", gcode_path,
            verified_stl_path
        ]
        proc = subprocess.run(slicer_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"PrusaSlicer error: {proc.stderr or proc.stdout}")

        grams_used = 0.0
        print_time_str = ""
        with open(gcode_path, "r") as f:
            for line in f.readlines()[-100:]:
                if "filament used [g]" in line:
                    try:
                        grams_used = float(line.split("=")[1].strip())
                    except Exception:
                        pass
                if "estimated printing time (normal mode)" in line:
                    print_time_str = line.split("=")[1].strip()

        # Резервна физичка пресметка од волумен (mm3 -> cm3 -> g) ако PrusaSlicer врати 0
        if grams_used <= 0.1:
            vol_cm3 = abs(float(final_mesh.volume)) / 1000.0
            grams_used = max(round(vol_cm3 * 1.24 * 0.45, 1), 1.5)

        if not print_time_str:
            est_mins = max(int(grams_used * 3.5), 12)
            hrs, mins = divmod(est_mins, 60)
            print_time_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

        rec_mat = ai_data.get("recommended_material", "PLA")
        material_cost = max(round(grams_used * MATERIAL_PRICES_MKD_PER_G.get(rec_mat, 1.5), 0), 10)
        selling_price = max(round(material_cost * 3, 0), 150)

        default_slicer = {
            "layer_height": "0.20 mm",
            "infill": "20% (Gyroid)",
            "walls": "3 периметри (1.2 mm)",
            "supports": "Auto (Touching buildplate)",
            "nozzle_temp": "210°C (PLA) / 240°C (PETG)",
            "bed_temp": "60°C (PLA) / 80°C (PETG)",
            "print_speed": "50 mm/s",
            "adhesion": "Brim 5 mm"
        }
        slicer_settings = ai_data.get("slicer_settings") or default_slicer

        return {
            "status": "ready",
            "model_id": model_id,
            "stl_url": f"/models/{model_id}_verified.stl",
            "message": ai_data["message"],
            "recommended_material": rec_mat,
            "material_reason": ai_data["material_reason"],
            "slicer_settings": slicer_settings,
            "stats": {
                "grams": round(grams_used, 1),
                "print_time": print_time_str,
                "material_cost_mkd": material_cost,
                "selling_price_mkd": selling_price,
                "is_watertight": bool(final_mesh.is_watertight),
                "dimensions_mm": [round(x, 1) for x in final_mesh.extents]
            }
        }
    except Exception as e:
        print("3D/SLICER ERROR:", traceback.format_exc())
        return {
            "status": "needs_info",
            "message": f"Техничка грешка при 3D генерација: {str(e)}"
        }

@app.post("/api/order")
async def create_order(order: OrderRequest):
    order_record = order.model_dump()
    order_record["verified_file"] = f"generated_models/{order.model_id}_verified.stl"
    with open("orders_database.jsonl", "a", encoding="utf-8") as db:
        db.write(json.dumps(order_record, ensure_ascii=False) + "\n")
    return {"status": "confirmed", "message": "Нарачката е примена! Проверениот STL фајл е подготвен за 3D печатач."}
