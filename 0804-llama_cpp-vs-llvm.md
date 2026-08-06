New Chat
========

[USER]
what's the best AI chat assistant that can run on truenas and has a ui with good search capabilities?

[ASSISTANT]
## Top pick: Open WebUI

For most TrueNAS users, **Open WebUI** is the best fit. It's available directly in the TrueNAS SCALE community apps catalog (as is Ollama for running models locally), and it's the most actively developed self-hosted chat UI.

**Why it stands out for search specifically:**
- **Chat history search** — quick search across all past conversations
- **Web search integration** — works with SearXNG (which you can also self-host on TrueNAS for fully private search), Tavily, Brave, DuckDuckGo, and others
- **Document RAG** — upload files or point it at collections, with hybrid search (BM25 + vector embeddings) and optional reranking
- **Multi-user support** with role-based access if others share your NAS

## Alternatives worth considering

| App             | Strengths                                                    |
| --------------- | ------------------------------------------------------------ |
| **LibreChat**   | Excellent conversation search via Meilisearch; best-in-class multi-provider support (OpenAI, Anthropic, local models in one UI) |
| **AnythingLLM** | Best if your main use case is chatting with large document collections; workspace-based organization; also in the TrueNAS catalog |
| **LobeChat**    | Polished UI, plugin ecosystem, but search features are weaker |

## Practical notes for TrueNAS

1. **Hardware reality check**: Most NAS CPUs will struggle with anything beyond small models (3B–8B parameters). Options:
   - Run small models like Llama 3.2 3B or Qwen 2.5 7B (quantized) on CPU — usable but slow
   - Add a GPU if your hardware allows (NVIDIA works well with SCALE)
   - Skip local inference entirely and point Open WebUI at an external API (OpenAI, Anthropic, Groq, OpenRouter) — the UI and search features work the same

2. **SCALE version matters**: On Electric Eel (24.10+), apps run on plain Docker, which makes custom deployments much easier than the old Kubernetes setup. Open WebUI and Ollama are both one-click installs from the community catalog.

3. **Suggested stack**: Open WebUI + Ollama + SearXNG, all from the catalog, gives you a fully private setup with web search in about 15 minutes.

**One caveat**: if by "search" you mainly meant *finding old conversations*, LibreChat's Meilisearch backend is arguably the strongest there — but Open WebUI's overall package is better for most people.

What's your hardware like? I can suggest specific models that would run well on it.

[USER]
what is the difference between vllm and llama-cpp?

[ASSISTANT]
## TL;DR

**llama.cpp** is built for running models efficiently on minimal hardware (CPU-first, single user). **vLLM** is built for serving models at high throughput to many concurrent users (GPU-first, production serving). They solve different problems.

## Side-by-side

|                     | **llama.cpp**                                 | **vLLM**                                    |
| ------------------- | --------------------------------------------- | ------------------------------------------- |
| **Primary target**  | CPU, edge devices, consumer GPUs              | NVIDIA datacenter/consumer GPUs (CUDA)      |
| **Model format**    | GGUF (its own format)                         | HuggingFace safetensors, GPTQ/AWQ/FP8       |
| **Key trick**       | Aggressive quantization (down to ~2–4 bits)   | PagedAttention + continuous batching        |
| **Concurrency**     | Weak — best for 1–2 users                     | Excellent — dozens of simultaneous requests |
| **Memory strategy** | Shrink the model to fit anywhere              | Manage KV cache efficiently at scale        |
| **Partial offload** | Yes — split layers between GPU and system RAM | Basically wants the whole model in VRAM     |
| **Deployment**      | Single binary, trivially simple               | Python stack, more moving parts             |
| **Hardware floor**  | Runs on a Raspberry Pi                        | Needs a real GPU with decent VRAM           |

## The core technical difference

**llama.cpp's approach: quantization.** It shrinks model weights from 16-bit down to 8, 4, or even 2 bits per weight. A 70B model that needs 140GB at full precision can fit in ~40GB at 4-bit. This makes big models *runnable* on ordinary hardware, at some cost to quality and speed.

**vLLM's approach: memory management and scheduling.** Its innovation (PagedAttention, from the UC Berkeley paper) treats the KV cache like an OS treats virtual memory — allocating it in pages instead of contiguous blocks, eliminating the huge waste in naive serving. Combined with continuous batching (new requests join the batch mid-flight as others finish), it squeezes dramatically more throughput out of the same GPU. The model itself usually stays at higher precision.

Put differently: llama.cpp asks *"how small can we make this model?"* while vLLM asks *"how many requests can we pump through this GPU?"*

## What this means for your TrueNAS setup

- **Ollama is llama.cpp under the hood** — so if you followed the earlier suggestion, you're already using it. That's the right call for a NAS: modest or no GPU, one user at a time, quantized GGUF models.
- **vLLM only makes sense if** you have a serious GPU (think 24GB+ VRAM), want to serve multiple users or apps simultaneously, and care about total throughput rather than single-request latency. It can also be faster per-request on big GPUs due to better kernel optimization.
- A common pattern: **llama.cpp/Ollama for personal use, vLLM if you later build something multi-user** (e.g., serving your household or a small team). Both expose OpenAI-compatible APIs, so Open WebUI can talk to either — switching backends later doesn't require changing your frontend.

One nuance: for a *single* request on consumer hardware, llama.cpp is often just as fast or faster than vLLM. vLLM's advantage only really shows up under concurrent load.

[USER]
does open webui work equally well on Android phone and desktop browser?

[ASSISTANT]
## Short answer

Yes, mostly — it's the same responsive web app on both, and Open WebUI ships with **PWA (Progressive Web App) support**, so on Android you can "Add to Home Screen" from Chrome and it behaves like an app: full-screen, its own icon, no browser chrome. Core chat, streaming, markdown/code rendering, model switching, and RAG all work identically since everything runs server-side anyway.

That said, the experience isn't perfectly equal:

## Where desktop is better

- **Admin and settings work** — the admin panel, user management, and pipeline/function configuration are usable on mobile but cramped
- **Document management for RAG** — uploading and organizing knowledge collections is much easier with a real file manager and drag-and-drop
- **Wide content** — big tables, side-by-side model comparisons, and long code blocks are more comfortable on a large screen
- **Keyboard shortcuts** — desktop-only in practice

## Where the phone is actually nicer

- **Voice input** — Open WebUI supports speech-to-text, and talking to your phone is more natural than a desktop mic
- **Camera/photo upload** — snap a photo of a document or whiteboard straight into a chat with a vision model
- **Server-side generation** — since inference happens on your NAS, you can start a long response, switch apps, and come back to it finished

## Practical tips for your TrueNAS setup

1. **Remote access**: On your LAN it just works, but for access away from home, use **Tailscale** (available as a TrueNAS app) rather than exposing Open WebUI to the internet. Install Tailscale on the NAS and your phone, and the PWA works from anywhere.
2. **Mic permissions**: If voice input misbehaves, check that Chrome has microphone permission for the site — and note that mic access generally requires HTTPS or a trusted context, which is another reason Tailscale (with its HTTPS support) helps.
3. **Use Chrome for the PWA install** — Firefox on Android has weaker PWA support.

One caveat: as far as I know there's no official *native* Open WebUI mobile app — the PWA is the intended mobile experience. There are unofficial wrappers floating around, but I'd stick with the PWA from your own server.

[USER]
does vllm use its own ui or does it use something like open webui

[ASSISTANT]
## vLLM has no UI at all — it's purely a backend

vLLM is an **inference engine only**. It serves models through an API and that's it. You always pair it with a separate frontend like Open WebUI.

The stack looks like this:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Open WebUI    │────▶│  Inference API   │────▶│   Model     │
│   (frontend)    │◀────│  (vLLM server)   │◀────│  weights    │
└─────────────────┘     └──────────────────┘     └─────────────┘
     Chat interface          OpenAI-compatible      Safetensors
     Search, RAG, users      /v1/chat/completions   on your GPU
```

## How they connect

vLLM runs as a server exposing an **OpenAI-compatible API** (`vllm serve <model>`). Open WebUI has built-in support for exactly this kind of endpoint:

1. In Open WebUI, go to **Settings → Admin Settings → Connections**
2. Add an OpenAI API connection pointing at your vLLM server (e.g., `http://your-nas:8000/v1`)
3. vLLM's models appear in Open WebUI's model dropdown alongside anything else

From the user's perspective it's indistinguishable from using Ollama — same chat UI, same search, same RAG features.

## Worth noting

- **Same is true of Ollama** — it also has no real chat UI (just a CLI and API). That's the whole reason Open WebUI exists as a separate project; it was originally built as a frontend *for* Ollama.
- **One frontend, many backends**: Open WebUI can connect to Ollama, vLLM, and cloud APIs (OpenAI, Anthropic, etc.) simultaneously, and you pick per-chat which backend/model to use. So if you ever add a GPU and stand up vLLM, you don't migrate anything — you just add another connection.
- vLLM does ship a basic demo via `vllm.entrypoints` examples, but it's a developer toy, not a usable product.

So the division of labor: **vLLM/llama.cpp = how fast the model runs; Open WebUI = how you interact with it.** They're complementary, not alternatives.

[USER]
is there any llama-cpp truenas app? i have an Nvidia rtx 4060ti 16gb vram on the truenas machine

[ASSISTANT]
## Direct answer

There's **no dedicated llama.cpp app in the TrueNAS community catalog** — but you have two good paths, and one of them is arguably better than a dedicated app anyway:

## Option 1: Ollama app (recommended)

**Ollama *is* llama.cpp** — it uses llama.cpp as its inference backend. So the Ollama app in the TrueNAS catalog already gives you llama.cpp's GGUF support, quantization, and CUDA acceleration, wrapped in a nicer management layer (model pulling, automatic GPU detection, OpenAI-compatible API). For your use case this is the pragmatic choice:

1. Install **Ollama** from the community catalog
2. In the app config, allocate your NVIDIA GPU (SCALE ships with NVIDIA drivers; there's a GPU section in the app settings)
3. Point Open WebUI at it (Open WebUI has native Ollama support)

## Option 2: llama.cpp server as a Custom App

If you want raw llama.cpp specifically (e.g., for flags Ollama doesn't expose, like specific KV cache quantization or sampling controls), Electric Eel's Docker-based Custom Apps make this easy. The project publishes official CUDA images:

```yaml
services:
  llama-cpp:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8080:8080"
    volumes:
      - /mnt/tank/models:/models
    command: >
      -m /models/mistral-small-3.2-24b-instruct-q4_k_m.gguf
      --host 0.0.0.0 --port 8080
      -ngl 99 --ctx-size 16384 --flash-attn
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Bonus: `llama-server` ships with its **own built-in web UI** at port 8080 — a basic but functional chat interface. You'd still want Open WebUI for search/RAG/multi-user, but it's handy for quick testing. It also exposes an OpenAI-compatible API, so Open WebUI connects to it the same way as vLLM.

## What your 4060 Ti 16GB can run

16GB VRAM is a genuinely good spot for local inference. Sweet spots:

| Model                       | Quant        | VRAM     | Notes                                                |
| --------------------------- | ------------ | -------- | ---------------------------------------------------- |
| **Mistral Small 3.2 (24B)** | Q4_K_M       | ~14.5GB  | Probably the best all-rounder that fits              |
| **gpt-oss-20b**             | native MXFP4 | ~12GB    | Fast, strong reasoning, lots of headroom for context |
| **Qwen3-30B-A3B (MoE)**     | Q3/IQ4       | ~15–16GB | Only ~3B active params → very fast tokens/sec        |
| **Gemma 3 12B**             | Q5/Q6        | ~10–12GB | Has vision (image input)                             |
| **Llama 3.1 8B**            | Q8           | ~9GB     | Leaves room for huge context                         |

Two hardware notes:

- The 4060 Ti's weak point is **memory bandwidth (288 GB/s)** — generation speed on a 24B Q4 will be roughly 15–25 tokens/sec. Fine for chat, not instant.
- Keep context sizes reasonable (8–16K) on the 24B models — the KV cache eats into your 16GB alongside the weights. Enabling `--flash-attn` (or q8_0 KV cache in Ollama via `OLLAMA_KV_CACHE_TYPE`) helps a lot.

**My suggestion**: start with the Ollama catalog app + Open WebUI. Only drop to raw llama.cpp if you hit a specific limitation — the day-to-day experience is identical.

[USER]
mxfp4 is apple hardware isnt it?

[ASSISTANT]
## No — MXFP4 isn't hardware at all, and Apple isn't involved

You're probably thinking of **MLX**, which *is* Apple's machine learning framework for Apple Silicon. The names are confusingly similar, but they're unrelated:

- **MLX** — Apple's array framework (their answer to PyTorch/NumPy) for M-series chips
- **MXFP4** — a 4-bit floating-point **data format** from the Open Compute Project's Microscaling (MX) specification, developed by a consortium of Microsoft, Meta, AMD, Arm, Intel, NVIDIA, and Qualcomm — notably *not* Apple

## What MXFP4 actually is

It's a quantization format: each weight is stored as a 4-bit float, with a shared scale factor per block of 32 values. The relevance to our discussion is that **OpenAI's gpt-oss models were trained with MXFP4 quantization baked in** for their MoE expert weights — that's why gpt-oss-20b ships at only ~12–13GB despite having 21B parameters.

## The hardware wrinkle that *does* affect you

Native FP4 acceleration in tensor cores is a **Blackwell** feature (RTX 50-series, B200) and AMD's newer Instinct cards. Your 4060 Ti (Ada Lovelace) does **not** have native FP4 hardware.

But that doesn't break anything for you: llama.cpp/Ollama handle MXFP4 weights in software on any CUDA GPU, dequantizing on the fly. You lose the speed bonus that an RTX 5090 would get, but gpt-oss-20b still runs well on a 4060 Ti — the MoE architecture (only ~3.6B active parameters per token) keeps generation fast regardless.

So my earlier recommendation stands: gpt-oss-20b is a good fit for your 16GB card, MXFP4 and all. The only scenario where the format would matter to you is if you were shopping for a Blackwell GPU to run it faster.

[USER]
I'm thinking of a hybrid approach. ollama takes to long to load models into memory. I'm thinking of ollama plus open webui for cloud models and llama-cpp with the strongest model always resident for OpenClaw home assistant MCP and lan callers like comfui prompt assistant

[ASSISTANT]
## That's a solid architecture — it's the classic "resident workhorse + on-demand + cloud" pattern

Your reasoning is sound: llama.cpp loads the model **once at container start** and holds it forever, which is exactly what always-on callers (OpenClaw, ComfyUI) need. Ollama's load-on-use behavior is fine for interactive chat where a 10–30 second cold start is tolerable... but you're right that it's annoying, and it's the wrong fit for API callers that expect instant responses.

```
                        ┌──────────────────────────────┐
                        │          Open WebUI          │
                        └──┬───────────┬───────────┬───┘
                           │           │           │
              ┌────────────▼─┐  ┌──────▼─────┐  ┌──▼──────────────┐
   Browser ──▶│  Cloud APIs  │  │   Ollama   │  │  llama-server   │
              │ (OpenAI etc) │  │ (on-demand)│  │  (RESIDENT)     │
              └──────────────┘  └────────────┘  └──┬───────┬──────┘
                                                   │       │
                                          ┌────────▼─┐  ┌──▼────────┐
                                          │ OpenClaw │  │ ComfyUI   │
                                          │ (HA MCP) │  │ prompt asst│
                                          └──────────┘  └───────────┘
```

One refinement: Open WebUI talks to cloud APIs **directly** (Settings → Connections) — Ollama isn't needed in that path at all. Ollama only earns its place for on-demand local models.

## The real design constraint: your 16GB VRAM budget

Both containers can share the GPU (NVIDIA allows this; VRAM is first-come-first-served), so you must decide how much headroom the resident model leaves:

| Strategy               | Resident model               | VRAM    | Headroom for Ollama                |
| ---------------------- | ---------------------------- | ------- | ---------------------------------- |
| **Max strength**       | Mistral Small 3.2 24B Q4_K_M | ~14.5GB | ~1GB — Ollama spills to RAM (slow) |
| **Balanced (my pick)** | gpt-oss-20b (MXFP4)          | ~13GB   | ~2.5GB — small models only         |
| **Generous headroom**  | Gemma 3 12B Q6_K             | ~10GB   | ~6GB — fits an 8B Q4 on GPU        |

**For your specific use case, gpt-oss-20b is the strongest candidate** — and it's not close. OpenClaw is an agent that lives or dies on **tool calling**, and gpt-oss was explicitly trained for agentic tool use. Its harmony format is natively supported by llama.cpp, and the MoE architecture (~3.6B active params) means fast responses even on your 4060 Ti's modest bandwidth. Mistral Small 24B is arguably smarter at pure chat, but it's worse at tools *and* leaves zero headroom.

## llama-server config for multi-client duty

Two flags matter a lot here:

```yaml
command: >
  -m /models/gpt-oss-20b-mxfp4.gguf
  --host 0.0.0.0 --port 8080
  -ngl 99
  --ctx-size 32768
  --parallel 2
  --flash-attn
  --jinja
```

- **`--parallel 2`** — enables concurrent slots with continuous batching, so OpenClaw and ComfyUI don't queue behind each other. Caveat: **context is divided across slots** (32K ÷ 2 = 16K each).
- **`--jinja`** — required for proper tool-call parsing with most model templates (gpt-oss works natively, but you'll want this if you swap models).
- **Context sizing matters more than you'd think**: OpenClaw's system prompt + tool definitions + conversation history is *large* — often 8–16K tokens before the user says anything. Don't go below 16K per slot.

## Ollama side: tame the settings

- `OLLAMA_MAX_LOADED_MODELS=1` — stop it hoarding multiple models
- `OLLAMA_KEEP_ALIVE=10m` — keeps a model warm through a chat session without squatting on VRAM forever
- Note that `KEEP_ALIVE=-1` would make Ollama models resident too — but that's incompatible with sharing 16GB, which is exactly why your split design is correct

## Practical gotchas

1. **Put the resident model on an SSD pool**, not your spinning array. A 13GB GGUF loading from HDDs is minutes; from SSD it's seconds. Since the whole point is killing load latency, don't sabotage container restarts.
2. **Graceful degradation is built in**: if Ollama loads something that doesn't fit in remaining VRAM, llama.cpp automatically partially offloads to system RAM. Slower, but nothing crashes.
3. **Honest caveat about OpenClaw**: a 20B local model handles routine Home Assistant commands fine, but complex multi-step tool chains are where local models most visibly trail cloud frontier models. Consider configuring OpenClaw to route to your resident model by default with a cloud fallback for hard tasks — Open WebUI/llama.cpp make that an endpoint swap, not a rebuild.
4. **ComfyUI prompt assistance is the lightest workload here** — short prompts, short outputs. It'll happily coexist in a parallel slot.

Want me to sketch the full docker-compose for the llama.cpp Custom App including the SSD dataset mount and GPU reservation?

[USER]
"Ollama only earns its place for on-demand local models." ollama has a great selection of cloud models in a central place like glm 5.2 and Kimi k3

[ASSISTANT]
## Fair point — but those are Ollama's *Turbo* cloud models, and they're a different thing than they appear

Ollama Turbo is a paid hosted service (~$20/mo Pro tier) where models like GLM-5 and Kimi K2 run on Ollama's servers, not your NAS. Your local Ollama instance is just a proxy in that scenario — the API calls go out to `ollama.com` with your API key.

That reframes the architecture a bit:

## Where Ollama Turbo genuinely earns its place

- **One connection, one bill** — GLM-5, Kimi K2, DeepSeek, Qwen3-Coder 480B, gpt-oss-120B all behind a single endpoint and subscription. That's real convenience versus juggling accounts at Zhipu, Moonshot, DeepSeek, etc.
- **Unified with local models** — same `ollama run` interface, same API, same Open WebUI connection. Switching between local gpt-oss-20b and cloud GLM-5 is a dropdown change.
- **Privacy caveat**: prompts leave your network and go through Ollama's infrastructure. For Home Assistant commands and personal chat, that may or may not matter to you.

## Where it's weaker than going direct

- **Per-model pricing/features can lag the source.** Moonshot and Zhipu sell API access directly, often cheaper per token than Turbo's bundled subscription if your usage is light.
- **No control over the backend.** You can't pick quant, context size, or provider — you get what Ollama configured.
- **Rate limits on the Pro tier** are real; heavy agentic use (OpenClaw doing long tool chains) can hit them.

## Revised architecture

```
                        ┌──────────────────────────────┐
                        │          Open WebUI          │
                        └──┬───────────┬───────────┬───┘
                           │           │           │
              ┌────────────▼─┐  ┌──────▼─────┐  ┌──▼──────────────┐
   Browser ──▶│  Cloud APIs  │  │   Ollama   │  │  llama-server   │
              │ (direct:     │  │ local +    │  │  (RESIDENT)     │
              │  Anthropic,  │  │ Turbo      │  │  gpt-oss-20b    │
              │  OpenRouter) │  │ cloud      │  └──┬───────┬──────┘
              └──────────────┘  └────────────┘     │       │
                                          ┌────────▼─┐  ┌──▼────────┐
                                          │ OpenClaw │  │ ComfyUI   │
                                          │ (HA MCP) │  │ prompt asst│
                                          └──────────┘  └───────────┘
```

Open WebUI happily merges all three paths into one model dropdown — it doesn't care whether a model is local GGUF, Ollama Turbo, or a direct cloud API. So your Ollama instance becomes the "Swiss Army knife" connection (local on-demand + Turbo cloud), while llama.cpp stays the always-on workhorse and direct APIs fill any gaps.

## One suggestion

If you're paying for Turbo anyway, **also set up one direct cloud connection** (OpenRouter is the easiest single key) as a fallback. When Turbo rate-limits you mid-conversation — and it will during heavy agent use — you can flip to the same model via a different provider without leaving Open WebUI.

[USER]
is OpenClaw like Claude code cli?

[ASSISTANT]
## Cousins, not twins — both are agent loops with tools, but built for different lives

**Claude Code** is a *developer tool*: an interactive terminal agent for working on a codebase — reading files, editing code, running tests, git operations. **OpenClaw** is an *always-on personal assistant gateway*: you talk to it through messaging apps (WhatsApp, Telegram, Discord, etc.) or its web UI, and it can act on your behalf — browse, run shell commands, control smart home gear, run scheduled tasks.

|                 | **Claude Code**                               | **OpenClaw**                                                 |
| --------------- | --------------------------------------------- | ------------------------------------------------------------ |
| **Primary job** | Software development                          | General-purpose agent / life automation                      |
| **Interface**   | Terminal CLI (+ IDE integrations)             | Messaging apps, web UI — async, multi-channel                |
| **Models**      | Anthropic Claude only                         | Model-agnostic — any OpenAI-compatible endpoint, including your llama-server |
| **Runtime**     | Interactive session in your project directory | Long-running background gateway, always on                   |
| **Trigger**     | You typing at it                              | Incoming messages, cron schedules, events — it can message *you* first |
| **Tooling**     | File editing, bash, git, MCP                  | Skills ecosystem, browser control, shell, MCP, messaging     |
| **Memory**      | `CLAUDE.md` + session context                 | Persistent workspace files, long-term memory, daily logs     |

## The differences that matter for your setup

1. **Model lock-in is the big one.** Claude Code only talks to Anthropic's API — it can never use your resident gpt-oss-20b. OpenClaw is explicitly designed to point at any OpenAI-compatible endpoint, which is exactly why it fits your architecture. Fair warning though: OpenClaw is developed and tuned primarily against Claude, and its own docs steer you there. Agentic performance on a 20B local model will be noticeably rougher — simpler tool schemas and shorter chains help.

2. **Interaction model.** Claude Code is synchronous — you sit with it while it works. OpenClaw is event-driven and asynchronous — it answers messages, fires scheduled jobs, and reacts to events while you're doing something else. *That's* the workload that justifies a resident model: requests arrive at random hours and need instant response, not a 30-second cold start.

3. **Both speak MCP**, so your Home Assistant MCP server works with either — but OpenClaw is the one designed to live permanently in that role. Claude Code with HA MCP would be more like "help me write an automation YAML," not "turn the lights off when I message you."

## One security note, since you're wiring it to Home Assistant

OpenClaw executes shell commands and is reachable via chat — that makes prompt injection a real attack vector (a malicious message or web page it reads could try to hijack it). The project has had real security scrutiny on this. Practical mitigations: keep the gateway LAN-only (Tailscale, like your Open WebUI), use its allowlisting so only *your* accounts can talk to it, and be thoughtful about which HA entities the MCP server exposes — locks and garage doors deserve a second thought.

So in your stack: OpenClaw = the always-on agent hanging off your resident llama.cpp model; Claude Code = a separate thing you'd run on your laptop when actually writing code, talking to Anthropic directly. They don't overlap much.