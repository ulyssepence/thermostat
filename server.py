import asyncio
import json
import os
import time
from collections import defaultdict
from math import exp, log
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import openai

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
MODEL = "accounts/fireworks/models/mixtral-8x22b-instruct"

client = openai.OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY,
)

generation_semaphore = asyncio.Semaphore(5)

ip_active: dict[str, int] = defaultdict(int)
ip_timestamps: dict[str, list[float]] = defaultdict(list)


def compute_entropy(top_logprobs: dict[str, float]) -> tuple[float, float]:
    if not top_logprobs:
        return 0.0, 0.0
    probs = [exp(lp) for lp in top_logprobs.values()]
    chosen_prob = probs[0] if probs else 0.0
    total = sum(probs)
    remaining = max(0.0, 1.0 - total)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * log(p)
    if remaining > 0:
        entropy -= remaining * log(remaining)
    return entropy, chosen_prob


def check_rate_limit(ip: str) -> str | None:
    if ip_active.get(ip, 0) >= 1:
        return "Already generating — wait for current generation to finish"
    now = time.time()
    timestamps = ip_timestamps[ip]
    ip_timestamps[ip] = [t for t in timestamps if now - t < 60]
    if len(ip_timestamps[ip]) >= 5:
        return "Rate limit: max 5 generations per minute"
    return None


async def generate_stream(prompt: str, temperatures: list[float], ip: str, *,
                          context_window: int | None = None):
    ip_active[ip] += 1
    ip_timestamps[ip].append(time.time())
    try:
        full_text = prompt
        generated_tokens: list[str] = []

        for i, temp in enumerate(temperatures):
            temp = max(0.01, min(temp, 2.0))

            if context_window and len(generated_tokens) > context_window:
                ctx_text = prompt + "".join(generated_tokens[-context_window:])
            else:
                ctx_text = full_text

            attempt = 0
            while True:
                try:
                    response = await asyncio.to_thread(
                        client.completions.create,
                        model=MODEL,
                        prompt=ctx_text,
                        max_tokens=1,
                        temperature=temp,
                        logprobs=5,
                    )
                    break
                except (openai.RateLimitError, openai.APIStatusError) as e:
                    status = getattr(e, 'status_code', 429)
                    if isinstance(e, openai.APIStatusError) and status not in (429, 503):
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        return
                    is_coldstart = status == 503
                    if not is_coldstart and attempt >= 5:
                        yield f"data: {json.dumps({'error': 'Rate limit exceeded after retries'})}\n\n"
                        return
                    wait = min(2 ** attempt, 16)
                    msg = f"Model scaling up, retrying in {wait}s..." if is_coldstart else f"Rate limited, retrying in {wait}s..."
                    yield f"data: {json.dumps({'retry': True, 'wait': wait, 'message': msg})}\n\n"
                    await asyncio.sleep(wait)
                    attempt += 1
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    return

            choice = response.choices[0]
            token_text = choice.text

            entropy, chosen_prob = 0.0, 0.0
            if choice.logprobs and choice.logprobs.top_logprobs:
                entropy, chosen_prob = compute_entropy(choice.logprobs.top_logprobs[0])
            full_text += token_text
            generated_tokens.append(token_text)

            if "<|" in token_text or choice.finish_reason == "stop":
                yield f"data: {json.dumps({'done': True, 'reason': 'eos'})}\n\n"
                return

            yield f"data: {json.dumps({'index': i, 'text': token_text, 'temperature': temp, 'entropy': entropy, 'chosen_probability': chosen_prob})}\n\n"

            await asyncio.sleep(0.1)

        yield f"data: {json.dumps({'done': True})}\n\n"
    finally:
        ip_active[ip] = max(0, ip_active[ip] - 1)


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    temperatures = body.get("temperatures", [])

    if not prompt or not temperatures:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'Missing prompt or temperatures'})}\n\n"]),
            media_type="text/event-stream",
        )

    if len(temperatures) > 200:
        temperatures = temperatures[:200]

    ip = request.client.host if request.client else "unknown"
    error = check_rate_limit(ip)
    if error:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': error})}\n\n"]),
            media_type="text/event-stream",
        )

    context_window = body.get("context_window")

    async def stream():
        async with generation_semaphore:
            async for chunk in generate_stream(
                prompt, temperatures, ip,
                context_window=context_window,
            ):
                yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def index():
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
