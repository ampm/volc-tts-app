import os
import re
import base64
import tempfile
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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOLC_TTS_TOKEN = os.getenv("VOLC_TTS_TOKEN", "")
VOICE_TYPE = os.getenv("VOICE_TYPE", "BV001_streaming")
MAX_LEN = int(os.getenv("MAX_LEN", "300"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

if not VOLC_TTS_TOKEN:
    raise RuntimeError("环境变量 VOLC_TTS_TOKEN 未设置")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
CHUNKS_ROOT = Path("chunks"); CHUNKS_ROOT.mkdir(exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

app = FastAPI(title="VolcEngine TTS Pipeline — Concurrency+Progress")
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
    except Exception:
        return safe_chunks_by_sentence(para, max_len)

def smart_split_full(text: str, max_len: int = 300) -> List[str]:
    chunks: List[str] = []
    for para in natural_paragraphs(text):
        if len(para) <= max_len:
            chunks.append(para)
        else:
            chunks.extend(ai_split_paragraph_local(para, max_len))
    return chunks


# -------- TTS（并发+重试） --------
async def volc_tts_async(text: str, idx: int, out_dir: Path, client_http: httpx.AsyncClient) -> Path:
    url = "https://openspeech.bytedance.com/api/v1/tts"
    payload = {
        "app": {"token": VOLC_TTS_TOKEN},
        "audio": {"voice_type": VOICE_TYPE, "encoding": "mp3"},
        "request": {"text": text}
    }
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client_http.post(url, json=payload, timeout=60)
            data = r.json()
            audio_b64 = data.get("data", {}).get("audio")
            if not audio_b64:
                raise RuntimeError(f"TTS返回异常: {str(data)[:200]}")
            audio_bytes = base64.b64decode(audio_b64)
            fn = out_dir / f"{idx}.mp3"
            fn.write_bytes(audio_bytes)
            return fn
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

def merge_mp3(files: List[Path], output_file: Path) -> None:
    combined = AudioSegment.empty()
    for f in files:
        combined += AudioSegment.from_file(f, format="mp3")
    combined.export(output_file, format="mp3")


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

    chunks = smart_split_full(text, MAX_LEN)
    total = len(chunks)
    await set_job(job_id, state="splitted", total=total, done=0, failed=0, message="开始TTS...", files=[])

    sem = asyncio.Semaphore(CONCURRENCY)
    results: List[Path] = [None] * total
    async with httpx.AsyncClient() as http_client:
        async def worker(i: int, content: str):
            idx = i + 1
            try:
                async with sem:
                    fn = await volc_tts_async(content, idx, job_dir, http_client)
                results[i] = fn
                st = await get_job(job_id)
                await set_job(job_id, done=st.get("done", 0) + 1)
            except Exception:
                st = await get_job(job_id)
                await set_job(job_id, failed=st.get("failed", 0) + 1)

        tasks = [asyncio.create_task(worker(i, c)) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)

    await set_job(job_id, message="TTS完成，开始合并...")
    ok_files = [p for p in results if p and p.exists()]
    if not ok_files:
        await set_job(job_id, state="error", message="TTS全部失败")
        return

    out_file = OUTPUT_DIR / f"{Path(filename).stem}_{job_id}_final.mp3"
    merge_mp3(ok_files, out_file)
    seg_file = out_file.with_suffix(".segments.txt")
    with seg_file.open("w", encoding="utf-8") as segf:
        for i, c in enumerate(chunks, 1):
            segf.write(f"[{i}]\n{c}\n\n")

    await set_job(job_id, state="done", result=str(out_file), message="完成")


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
