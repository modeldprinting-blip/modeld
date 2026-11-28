import os
import re
import json
import uuid
import math
import asyncio
import subprocess
import traceback
import urllib.request
import urllib.error
import trimesh
import numpy as np
from groq import AsyncGroq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Model D — AI 3D CAD & Slicer Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("generated_models", exist_ok=True)
app.mount("/models", StaticFiles(directory="generated_models"), name="models")

GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
TRIPO_API_KEY = (os.environ.get("TRIPO_API_KEY") or "").strip()

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Директен HTTP opener без системски proxy променливи
no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def http_json_request(url: str, headers: dict, payload: dict | None = None, timeout: int = 60) -> dict:
    data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with no_proxy_opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_body}")

def http_download_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ModelD/1.0"})
    with no_proxy_opener.open(req, timeout=timeout) as resp:
        return resp.read()

MATERIAL_PRICES_MKD_PER_G = {
    "PLA": 1.5,
    "PETG": 1.8,
    "ABS": 1.7,
    "TPU": 2.6
}

PURE_CAD_KEYWORDS = [
    "kocka", "коцка", "kvadar", "квадар", "prsten", "прстен",
    "cilindar", "цилиндар", "plocka", "плочка", "cevka", "цевка",
    "konus", "конус", "podloska", "подлошка", "kapak", "капак"
]

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
- Користи `"mode": "tripo"` за 95% од предметите: СЕКОЈ реален, ергономски, уметнички, органски или детален објект (држач за телефон, новогодишна елка, украс, фигура, животно, скулптура, статуа, играчка, карактер, цвет, автомобил, сноуборд везови, маска, рачка, кацига, вазна, чаша со релјеф, резервен дел).
- Користи `"mode": "cad"` ИСКЛУЧИВО за наједноставна геометрија од 1 форма: обичен прстен, цилиндар, коцка, рамна плочка, цевка, конус.

КРИТИЧНИ ПРАВИЛА ЗА `tripo_prompt`:
- Дури и ако корисникот напише долг текст со барања за STEP фајлови, технички цртежи, документација или милиметри, во `tripo_prompt` НИКОГАШ не ги вклучувај тие зборови и НИКОГАШ не ставај бројки!
- Напиши САМО краток, чист визуелен опис на самиот 3D објект во 10-20 зборови на англиски (пр. "A single 3D printable ergonomic snowboard binding with curved highback heel support, flat baseplate, and side strap loops").
- Во `cad_script` напиши ИСКЛУЧИВО краток, 100% валиден Python код со `trimesh` кој ја дефинира `final_mesh`.
- Во `target_height_mm` стави ја висината во милиметри (минимум 15.0, максимум 220.0).

Врати JSON со ТОЧНО овие имиња на полиња:
{
  "status": "ready",
  "mode": "tripo" | "cad",
  "message": "Потврда на македонски дека моделот е физички проверен, поправен (Watertight) и подготвен за 3D печатење.",
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
  "tripo_prompt": "A single 3D printable solid object on a flat base, clean geometry, no floating parts, untextured white clay model: [краток визуелен опис на англиски]",
  "target_height_mm": 100.0,
  "cad_script": "final_mesh = trimesh.creation.box(extents=[80.0, 80.0, 100.0])"
}
Одговори ИСКЛУЧИВО со валиден JSON објект, без никаков дополнителен текст пред или после него.
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
    return {
        "status": "online",
        "service": "Model D Backend",
        "tripo_enabled": bool(TRIPO_API_KEY),
        "groq_enabled": bool(GROQ_API_KEY)
    }

@app.get("/test-tripo")
async def test_tripo_connection():
    if not TRIPO_API_KEY:
        return {"ok": False, "error": "TRIPO_API_KEY не е поставен во Railway Variables!"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TRIPO_API_KEY}"
    }
    try:
        bal = await asyncio.to_thread(
            http_json_request,
            "https://api.tripo3d.ai/v2/openapi/user/balance",
            headers,
            None,
            25
        )
        task = await asyncio.to_thread(
            http_json_request,
            "https://api.tripo3d.ai/v2/openapi/task",
            headers,
            {"type": "text_to_model", "prompt": "a simple small star ornament"},
            25
        )
        return {
            "ok": task.get("code") == 0,
            "balance": bal,
            "task": task
        }
    except Exception as e:
        return {"ok": False, "exception": str(e)}

def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)

async def call_groq(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        role = "assistant" if m.get("role") in ["assistant", "model", "ai"] else "user"
        messages.append({"role": role, "content": m.get("content", "")})

    groq_models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant"
    ]
    last_error = None

    for model_name in groq_models:
        try:
            completion = await groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=4096
            )
            content = (completion.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            print("GROQ MODEL RETRY:", last_error)

    raise RuntimeError(last_error)

async def generate_with_tripo(prompt: str, output_stl_path: str, target_height_mm: float):
    clean_prompt = str(prompt).strip()[:300]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TRIPO_API_KEY}"
    }
    res_data = await asyncio.to_thread(
        http_json_request,
        "[https://api.tripo3d.ai/v2/openapi/task](https://api.tripo3d.ai/v2/openapi/task)",
        headers,
        {"type": "text_to_model", "prompt": clean_prompt},
        30
    )
    if res_data.get("code") != 0:
        raise RuntimeError(f"Tripo API одби ({res_data.get('code')}): {res_data.get('message')}")
    task_id = res_data["data"]["task_id"]

    model_url = None
    for _ in range(60):
        await asyncio.sleep(2)
        status_res = await asyncio.to_thread(
            http_json_request,
            f"[https://api.tripo3d.ai/v2/openapi/task/](https://api.tripo3d.ai/v2/openapi/task/){task_id}",
            headers,
            None,
            25
        )
        s_data = status_res.get("data", {})
        st = s_data.get("status")
        if st == "success":
            out = s_data.get("output", {})
            model_url = out.get("model") or out.get("pbr_model") or out.get("base_model")
            break
        elif st in ["failed", "cancelled", "banned"]:
            raise RuntimeError(f"Tripo статус: {st}")

    if not model_url:
        raise RuntimeError("Tripo истече времето (timeout).")

    glb_path = output_stl_path.replace(".stl", ".glb")
    glb_bytes = await asyncio.to_thread(http_download_bytes, model_url, 90)
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)

    loaded = trimesh.load(glb_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    loaded.apply_transform(rot_x)

    current_height = loaded.extents[2] if loaded.extents[2] > 0 else 1.0
    clamped_height = min(max(target_height_mm, 15.0), 220.0)
    scale_factor = clamped_height / current_height
    loaded.apply_scale(scale_factor)

    max_xy = max(loaded.extents[0], loaded.extents[1])
    if max_xy > 230.0:
        loaded.apply_scale(230.0 / max_xy)

    return loaded

def run_cad_script(cad_script: str):
    exec_globals = {
        "__builtins__": __builtins__,
        "trimesh": trimesh,
        "np": np,
        "numpy": np,
        "math": math
    }
    exec(cad_script, exec_globals)
    if "final_mesh" in exec_globals:
        return exec_globals["final_mesh"]
    for val in reversed(list(exec_globals.values())):
        if isinstance(val, (trimesh.Trimesh, trimesh.Scene)):
            return val
    raise RuntimeError("CAD скриптата не врати 3D објект.")

@app.post("/api/generate")
async def generate_3d(req: ChatRequest):
    if not groq_client:
        return {"status": "needs_info", "message": "Грешка: Нема поставено GROQ_API_KEY на серверот."}

    try:
        raw_text = await call_groq(req.history)
    except Exception as e:
        print("GROQ ERROR:", traceback.format_exc())
        return {"status": "needs_info", "message": f"Groq AI грешка: {str(e)}"}

    try:
        ai_data = extract_json_object(raw_text)
    except Exception as e:
        print("RAW AI TEXT:", raw_text)
        return {"status": "needs_info", "message": f"AI JSON грешка: {str(e)}"}

    if ai_data.get("status") == "needs_info":
        return ai_data

    model_id = str(uuid.uuid4())[:8]
    os.makedirs("generated_models", exist_ok=True)
    verified_stl_path = f"generated_models/{model_id}_verified.stl"
    gcode_path = f"generated_models/{model_id}.gcode"

    try:
        final_mesh = None
        tripo_debug_note = ""
        full_user_text = " ".join([m.get("content", "").lower() for m in req.history if m.get("role") == "user"])
        is_pure_simple_cad = any(kw in full_user_text for kw in PURE_CAD_KEYWORDS) and len(full_user_text) < 90

        tripo_prompt = (ai_data.get("tripo_prompt") or ai_data.get("prompt") or "").strip()
        if not tripo_prompt:
            tripo_prompt = "A single 3D printable ergonomic object on a flat solid base, clean geometry, white clay model"

        use_tripo = bool(TRIPO_API_KEY) and (ai_data.get("mode") == "tripo" or not is_pure_simple_cad)

        if use_tripo:
            try:
                final_mesh = await generate_with_tripo(
                    tripo_prompt,
                    verified_stl_path,
                    float(ai_data.get("target_height_mm") or 80.0)
                )
            except Exception as tripo_err:
                tripo_debug_note = f" (Tripo забелешка: {str(tripo_err)})"
                print("TRIPO FALLBACK:", traceback.format_exc())

        if final_mesh is None:
            script_code = (
                ai_data.get("cad_script")
                or ai_data.get("script")
                or ai_data.get("code")
                or ai_data.get("python_code")
                or ""
            ).strip()

            if script_code:
                try:
                    final_mesh = run_cad_script(script_code)
                except Exception:
                    print("CAD FALLBACK:", traceback.format_exc())

            if final_mesh is None:
                h = min(max(float(ai_data.get("target_height_mm") or 60.0), 15.0), 200.0)
                if "konus" in full_user_text or "конус" in full_user_text or "elka" in full_user_text or "елка" in full_user_text:
                    final_mesh = trimesh.creation.cone(radius=h * 0.35, height=h)
                elif "cilindar" in full_user_text or "цилиндар" in full_user_text or "casa" in full_user_text or "чаша" in full_user_text:
                    final_mesh = trimesh.creation.cylinder(radius=h * 0.4, height=h)
                else:
                    final_mesh = trimesh.creation.box(extents=[h, h * 0.6, max(h * 0.3, 10.0)])

        if isinstance(final_mesh, trimesh.Scene):
            final_mesh = trimesh.util.concatenate(tuple(final_mesh.geometry.values()))

        if len(final_mesh.vertices) == 0 or max(final_mesh.extents) <= 0.1:
            final_mesh = trimesh.creation.box(extents=[50.0, 50.0, 20.0])

        try:
            trimesh.repair.fill_holes(final_mesh)
            trimesh.repair.fix_normals(final_mesh)
        except Exception as repair_err:
            print("MESH REPAIR WARNING:", str(repair_err))

        final_mesh.apply_translation([
            -final_mesh.centroid[0],
            -final_mesh.centroid[1],
            -final_mesh.bounds[0][2]
        ])
        final_mesh.export(verified_stl_path)

        slicer_cmd = [
            "prusa-slicer",
            "--export-gcode",
            "--center", "100,100",
            "--support-material",
            "--support-material-auto",
            "--layer-height", "0.2",
            "--fill-density", "20%",
            "-o", gcode_path,
            verified_stl_path
        ]
        proc = subprocess.run(slicer_cmd, capture_output=True, text=True)

        grams_used = 0.0
        print_time_str = ""
        if proc.returncode == 0 and os.path.exists(gcode_path):
            with open(gcode_path, "r") as f:
                for line in f.readlines()[-100:]:
                    if "filament used [g]" in line:
                        try:
                            grams_used = float(line.split("=")[1].strip())
                        except Exception:
                            pass
                    if "estimated printing time (normal mode)" in line:
                        print_time_str = line.split("=")[1].strip()

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

        base_msg = ai_data.get("message", "Моделот е подготвен за 3D печатење!")
        return {
            "status": "ready",
            "model_id": model_id,
            "stl_url": f"/models/{model_id}_verified.stl",
            "message": base_msg + tripo_debug_note,
            "recommended_material": rec_mat,
            "material_reason": ai_data.get("material_reason", "Оптимален материјал според намената и оптоварувањето."),
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
