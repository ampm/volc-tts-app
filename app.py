import os
import re
import json
import base64
import tempfile
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import asyncio
import httpx
from pydub import AudioSegment

# 可选：OpenAI，用于AI语义分段（没有Key会自动回退到安全切分）
from openai import OpenAI

# ========== ENV ==========
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOLC_TTS_TOKEN = os.getenv("VOLC_TTS_TOKEN", "")
VOICE_TYPE = os.getenv("VOICE_TYPE", "BV001_streaming")
MAX_LEN = int(os.getenv("MAX_LEN", "300"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
VOLC_APP_ID = os.getenv("VOLC_APP_ID", "")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
def get_volc_tts_token(app_id, access_key_id, secret_access_key):
    url = "https://openspeech.bytedance.com/api/v1/token"
    payload = {
        "app_id": app_id,
        "access_key_id": access_key_id,
        "secret_access_key": secret_access_key,
        "expire_time": 3600
    }
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "token" not in data:
        raise RuntimeError(f"获取Token失败: {data}")
    return data["token"]

# 在 load_dotenv() 后调用
if not VOLC_TTS_TOKEN:
    VOLC_AK = os.getenv("VOLC_ACCESS_KEY_ID", "")
    VOLC_SK = os.getenv("VOLC_SECRET_ACCESS_KEY", "")
    if not (VOLC_APP_ID and VOLC_AK and VOLC_SK):
        raise RuntimeError("缺少 VOLC_APP_ID / VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY")
    VOLC_TTS_TOKEN = get_volc_tts_token(VOLC_APP_ID, VOLC_AK, VOLC_SK)
if not VOLC_TTS_TOKEN:
    raise RuntimeError("环境变量 VOLC_TTS_TOKEN 未设置")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ========== LOGGING ==========
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("volc-tts")

def job_log_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}.log"

def log_job(job_id: str, message: str):
    """写stdout & job专属日志文件"""
    logger.info(f"[{job_id}] {message}")
    lp = job_log_path(job_id)
    with lp.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {message}\n")

# ========== PATHS / STATE ==========
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
CHUNKS_ROOT = Path("chunks"); CHUNKS_ROOT.mkdir(exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

app = FastAPI(title="VolcEngine TTS Pipeline — Concurrency+Progress+Debug")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -------- 文本切分 --------
def natural_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n|\n|\r\n", text)]
    return [p for p in parts if p]

def safe_chunks_by_sentence(para: str, max_len: int = 300) -> List[str]:
    sents = re.split(r"([。！？；!?\u3002\uFF01\uFF1F\uFF1B])", para)
    sents = [s for s in sents if s and s.strip()]
    chunks, cur = [], ""
    for seg in sents:
        if len(cur) + len(seg) <= max_len:
            cur += seg
        else:
            if cur:
                chunks.append(cur.strip())
            if len(seg) <= max_len:
                cur = seg
            else:
                # 兜底硬切，极端长句
                for i in range(0, len(seg), max_len):
                    piece = seg[i:i+max_len]
                    if i == 0:
                        cur = piece
                    else:
                        chunks.append(cur.strip()); cur = piece
    if cur: chunks.append(cur.strip())
    return chunks

def ai_split_paragraph_local(para: str, max_len: int = 300) -> List[str]:
    if not client:
        return safe_chunks_by_sentence(para, max_len)
    prompt = (
        f"请将以下文本按语义自然分段，每段不超过{max_len}个汉字；"
        f"不要拆开句子；保持原文顺序；仅输出分段结果，每段独立一行：\n\n{para}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        lines = [re.sub(r"^\s*(\d+[\.\)、）])\s*", "", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        fixed = []
        for ln in lines:
            if len(ln) <= max_len:
                fixed.append(ln)
            else:
                fixed.extend(safe_chunks_by_sentence(ln, max_len))
        return fixed or safe_chunks_by_sentence(para, max_len)
    except Exception as e:
        logger.warning(f"AI分段失败，回退安全切分：{e}")
        return safe_chunks_by_sentence(para, max_len)

def smart_split_full(text: str, max_len: int = 300) -> List[str]:
    chunks: List[str] = []
    for para in natural_paragraphs(text):
        if len(para) <= max_len:
            chunks.append(para)
        else:
            chunks.extend(ai_split_paragraph_local(para, max_len))
    return chunks


# -------- TTS（并发+重试+日志） --------
async def volc_tts_async(text: str, idx: int, out_dir: Path, client_http: httpx.AsyncClient, job_id: str) -> Path:
    url = "https://openspeech.bytedance.com/api/v1/tts"
    payload = {
        "app": {
            "appid": VOLC_APP_ID,
            "token": VOLC_TTS_TOKEN,
            "cluster": "volcano_tts",
        },
        "user": {
            "uid": "uid123"
        },
        "audio": {"voice_type": VOICE_TYPE, "encoding": "mp3"},
         "request": {
            "reqid": str(uuid.uuid1()),
            "text": text,
            "operation": "query"
        }
    }
    
    headers = {
        "Authorization": f"Bearer; {VOLC_TTS_TOKEN}", 
        "Content-Type": "application/json",
    }

    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client_http.post(url, headers=headers, json=payload, timeout=60)
            status = r.status_code
            ctype = r.headers.get("Content-Type", "")
            body_preview = (r.text[:200] if r.text else str(r.content[:200]))
            log_job(job_id, f"TTS idx={idx} attempt={attempt} status={status} ctype={ctype} preview={body_preview!r}")

            if status != 200:
                raise RuntimeError(f"http {status}: {body_preview}")

            audio_bytes = None

            if "audio/" in ctype.lower():
                # 直接返回了音频二进制
                audio_bytes = r.content
            else:
                # JSON 或 文本
                data_obj = None
                if "application/json" in ctype.lower():
                    try:
                        data_obj = r.json()
                    except Exception as e:
                        log_job(job_id, f"TTS idx={idx} json parse failed: {e}. fallback=text")
                        data_obj = None

                audio_b64 = None

                if isinstance(data_obj, dict):
                    # 🔴 关键分支：data 可能是 str（直接base64）或 dict（里边才有 audio）
                    if "data" in data_obj:
                        data_field = data_obj["data"]
                        if isinstance(data_field, str):
                            # 你的返回就是这种：code=3000, data="<base64>"
                            audio_b64 = data_field
                        elif isinstance(data_field, dict):
                            audio_b64 = (
                                data_field.get("audio")
                                or data_field.get("Audio")
                                or data_field.get("result")
                            )
                    # 打点一下 keys 方便排查
                    try:
                        log_job(job_id, f"TTS idx={idx} json keys: {list(data_obj.keys())}")
                    except Exception:
                        pass

                # 如果还没拿到，尝试把整个文本当 base64（有些接口就直接回纯文本）
                if not audio_b64:
                    text_body = (r.text or "").strip().strip('"').strip("'")
                    if len(text_body) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', text_body or ""):
                        audio_b64 = text_body

                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception as e:
                        raise RuntimeError(f"base64 decode failed: {e}. head={audio_b64[:32]}...")

            if not audio_bytes:
                raise RuntimeError("no audio bytes parsed from response")

            fn = out_dir / f"{idx}.mp3"
            fn.write_bytes(audio_bytes)
            return fn

        except Exception as e:
            err_msg = f"TTS idx={idx} attempt={attempt} error: {e}"
            log_job(job_id, err_msg)

            # 失败详情文件
            fail_file = out_dir / f"{idx}_error.json"
            try:
                fail_payload = {
                    "idx": idx,
                    "attempt": attempt,
                    "error": str(e),
                    "response": {
                        "status": status if 'status' in locals() else None,
                        "content_type": ctype if 'ctype' in locals() else None,
                        "body_preview": body_preview if 'body_preview' in locals() else None,
                    },
                    "payload_preview": {
                        "voice_type": VOICE_TYPE,
                        "text_len": len(text),
                    }
                }
                fail_file.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            if attempt >= MAX_RETRIES:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

    raise RuntimeError("unexpected")


def merge_mp3(files: List[Path], output_file: Path) -> None:
    try:
        combined = AudioSegment.empty()
        for f in files:
            combined += AudioSegment.from_file(f, format="mp3")
        combined.export(output_file, format="mp3")
    except FileNotFoundError as e:
        # 多半是系统未安装 ffmpeg/ffprobe
        raise RuntimeError("合并失败：系统缺少 ffmpeg/ffprobe，请安装后重试") from e
    except Exception as e:
        raise RuntimeError(f"合并失败：{e}") from e


# -------- Job 状态 --------
async def set_job(job_id: str, **kwargs):
    async with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kwargs)

async def get_job(job_id: str) -> Dict[str, Any]:
    async with JOBS_LOCK:
        return JOBS.get(job_id, {}).copy()

async def process_job(job_id: str, filename: str, text: str):
    job_dir = CHUNKS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    log_job(job_id, f"Start process file={filename}, text_len={len(text)}")
    chunks = smart_split_full(text, MAX_LEN)
    total = len(chunks)
    await set_job(job_id, state="splitted", total=total, done=0, failed=0, message="开始TTS...", files=[], failures=[])
    log_job(job_id, f"Splitted chunks={total}")

    sem = asyncio.Semaphore(CONCURRENCY)
    results: List[Path] = [None] * total

    async with httpx.AsyncClient() as http_client:
        async def worker(i: int, content: str):
            idx = i + 1
            try:
                async with sem:
                    fn = await volc_tts_async(content, idx, job_dir, http_client, job_id=job_id)
                results[i] = fn
                st = await get_job(job_id)
                await set_job(job_id, done=st.get("done", 0) + 1)
            except Exception as e:
                st = await get_job(job_id)
                failures = st.get("failures", [])
                failures.append({"idx": idx, "error": str(e)})
                await set_job(job_id, failed=st.get("failed", 0) + 1, failures=failures)
                log_job(job_id, f"FAILED idx={idx}: {e}")

        tasks = [asyncio.create_task(worker(i, c)) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)

    await set_job(job_id, message="TTS完成，开始合并...")
    log_job(job_id, "Start merging...")

    ok_files = [p for p in results if p and p.exists()]
    if not ok_files:
        await set_job(job_id, state="error", message="TTS全部失败", log=str(job_log_path(job_id)))
        log_job(job_id, "ERROR: all segments failed")
        return

    out_file = OUTPUT_DIR / f"{Path(filename).stem}_{job_id}_final.mp3"
    try:
        merge_mp3(ok_files, out_file)
    except Exception as e:
        await set_job(job_id, state="error", message=str(e), log=str(job_log_path(job_id)))
        log_job(job_id, f"ERROR during merge: {e}")
        return
    
    seg_file = out_file.with_suffix(".segments.txt")
    with seg_file.open("w", encoding="utf-8") as segf:
        for i, c in enumerate(chunks, 1):
            segf.write(f"[{i}]\n{c}\n\n")

    await set_job(job_id, state="done", result=str(out_file), message="完成", log=str(job_log_path(job_id)))
    log_job(job_id, f"Done. output={out_file}")


# -------- 页面与API --------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/tts")
async def tts(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="请上传 .txt 文本文件")
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except Exception:
            raise HTTPException(status_code=400, detail="文本编码无法识别，请用UTF-8或GBK。")

    job_id = next(tempfile._get_candidate_names())
    await set_job(job_id, state="queued", total=0, done=0, failed=0, message="排队中...")
    log_job(job_id, f"Queued job for filename={file.filename}")
    asyncio.create_task(process_job(job_id, file.filename, text))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    st = await get_job(job_id)
    if not st:
        return JSONResponse({"error": "job_id 不存在"}, status_code=404)
    return st

@app.get("/result/{job_id}")
async def result(job_id: str):
    st = await get_job(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job_id 不存在")
    if st.get("state") != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    return FileResponse(st["result"], media_type="audio/mpeg", filename=Path(st["result"]).name)

# 新增：调试接口
@app.get("/debug/{job_id}")
async def debug(job_id: str):
    st = await get_job(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job_id 不存在")

    lp = job_log_path(job_id)
    log_tail = ""
    if lp.exists():
        # 只读最后20KB，避免太大
        data = lp.read_text(encoding="utf-8")[-20_000:]
        log_tail = data

    job_dir = CHUNKS_ROOT / job_id
    fail_files = sorted([str(p) for p in job_dir.glob("*_error.json")])

    return {
        "state": st.get("state"),
        "message": st.get("message"),
        "failures": st.get("failures", []),
        "log_file": str(lp),
        "log_tail": log_tail,
        "fail_files": fail_files,
        "output": st.get("result"),
    }
