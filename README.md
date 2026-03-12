# Thermostat

A web app that lets you define a temperature function over the output token index, then streams tokens from an LLM colored by temperature with an interactive line graph showing temperature and entropy curves.

You write a math expression like `stay01(i / 10, 6, 0, 8) * 2.0` where `i` is the token index. The app evaluates it to produce a temperature for each token, then generates one token at a time via the Fireworks completions API, streaming results back with colored highlighting and a live graph.

## How it works

1. The browser evaluates your temperature function with [math.js](https://mathjs.org/) to produce an array of temperatures
2. It POSTs the prompt + temperatures array to the server
3. The server calls Fireworks' completions API one token at a time, each with its own temperature, accumulating the text as context
4. Tokens stream back via SSE with temperature, entropy, and token probability
5. The browser renders tokens with temperature-colored backgrounds and updates a Canvas 2D graph

A **sliding context window** (default: 10 tokens) limits how much generated text the model sees as context, which makes high-temperature regions more chaotic since the model can't anchor on earlier coherent text.

## Running locally

```
cp .env.example .env  # add your FIREWORKS_API_KEY
uv run uvicorn server:app --host 127.0.0.1 --port 8080 --reload
```

## Docker

```
docker build -t thermostat .
docker run -p 8080:8080 -e FIREWORKS_API_KEY=fw_... thermostat
```

## Custom math functions

Click "Temperature function reference" in the UI for the full list. In addition to all built-in math.js functions, these are available:

| Function | Description |
|---|---|
| `sin01(x)` | Sine normalized to 0-1 |
| `triangle(t)` | Triangle wave |
| `square(t)` | Square wave |
| `stay01(t, period, offset, power)` | Wave that holds at extremes |
| `smoothstep(lo, hi, x)` | Smooth 0-to-1 transition |
| `pulse(center, width, x)` | Smooth bump |
| `noise(x)` | Deterministic pseudo-random |
| `map(v, lo1, hi1, lo2, hi2)` | Remap between ranges |
| `mix(a, b, t)` | Linear interpolation |
| `clamp(v, lo, hi)` | Clamp to range |
| `fract(v)` | Fractional part (sawtooth) |
| `snap(v, step)` | Quantize to grid |
| `step(edge, x)` | Heaviside step |
| `smin(a, b, k)` | Smooth minimum |
