# TensorScope — Feature Spec

> Living reference doc. Update as the project evolves — don't re-explain context in new chats, just link/paste this.

---

## 1. Vision

TensorScope is a **visual ML sketchpad**: assemble architecture blocks, see shapes validate live, and (eventually) run real inference to see what's happening inside a model — all before writing a single line of training code.

It exists to collapse the *long feedback loop* problem: idea → code → train → wait → evaluate → realize it was wrong → repeat. Each loop costs GPU hours. TensorScope lets you validate architectural intuition first, locally, instantly — then commit to training only once the idea is sound.

It is also, deliberately, a **teaching artifact**. Per the "learn so you can teach" principle: every ML concept learned gets built into the tool as a feature. The act of building it is the act of understanding it. By the time the platform is "done," it will double as an interactive beginner's guide to neural networks.

**Target users:** not just developers. ML students, researchers communicating architecture, and non-coders who think visually — all should be able to use it.

---

## 2. Current State (v1 — working prototype)

Built as a React artifact. Dark-themed, color-coded by layer type, monospace shape labels.

**Implemented:**
- Layer types: `Input`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dense`, `Reshape`, `GlobalAvgPool`, `BatchNorm`, `Dropout`, `ReLU`, `Softmax`, `Embedding`
- Live shape computation engine — propagates shape through the stack on every edit
- Plain-English error messages on shape mismatches (not stack traces)
- Click-to-select layer → inline param editor (filters, kernel size, stride, padding, units, etc.)
- Add / remove layers dynamically
- Mini spatial preview (`ShapeViz`) — a small proportional rectangle showing H×W per layer, with a corner marker for channel dimension
- Sidebar layer list with global error/valid status
- Final model output badge with total element count

**Known fixed bugs:**
- Click-event propagation bug where clicking an input field bubbled up and deselected the layer (fixed via `stopPropagation`)

**Known limitation:** mobile (phone) input editing is cramped — desktop is the primary target for now.

---

## 3. Core Design Principle

**Shape is the only hard contract.** TensorScope doesn't care about framework (PyTorch vs TensorFlow — pure math, framework-agnostic), doesn't validate semantics, doesn't know what a channel "means." It only enforces that shapes connect correctly between layers. This is intentional — it mirrors how real deployment failures actually happen (shape mismatches), and keeps the tool simple and universally applicable.

Mixing layers/blocks from different architectures (e.g., a Ghost module between two MobileNet blocks) is fully supported as long as shapes match at the junction. Hybrid architectures are the norm in real ML (EfficientDet, U-Net, etc.) — TensorScope should reflect that, not fight it.

---

## 4. Roadmap — 8-Stage Curriculum ↔ Feature Map

Each learning stage ships one concrete feature. Don't study ML separately from this — make the project the study.

| Stage | Concept to learn | Feature to build |
|---|---|---|
| 1 | What is a tensor (shape, dims) | Hover labels: "batch · height · width · channels" |
| 2 | What Dense does (weights, matmul) | Tooltip: "multiplies {in} inputs by {units} learned weights" |
| 3 | Why activations exist (ReLU, Softmax, Sigmoid) | Activation explainer cards with inline function-shape graph |
| 4 | Conv2D deep dive (filters, kernels, feature maps) | "What this layer detects" annotation — edges early, shapes deeper |
| 5 | Pooling + Flatten (why shrink, why flatten) | "Computation saved" stat on pooling; visual "unrolling" animation on flatten |
| 6 | Full forward pass (data flow start to finish) | "Trace" mode — click Run, watch shape animate through every layer |
| 7 | Model families (CNN vs Transformer vs RNN) | Domain selector (CV / NLP / Audio / Tabular) — unlocks only relevant layer palette |
| 8 | Famous architectures (VGG, MobileNet, BERT) | Preset templates with design annotations (e.g. "Depthwise Conv here saves 8× compute") |

**Status:** Currently mid-curriculum (3Blue1Brown NN fundamentals → backpropagation). Stages above not yet built — sequenced to follow conceptual understanding, not precede it.

---

## 5. Platform Vision (post-curriculum)

The longer-term product, once Stage 1–8 foundations are in place:

### 5.1 Interactive Model Builder
- "Start from scratch" flow: pick a domain first (📷 Image / 💬 Text / 🔊 Audio / 📊 Tabular)
- Only relevant layer types unlock per domain — no Conv2D showing up in an NLP build
- Preset templates, one click to load: MobileNetV2, VGG-16, Tiny BERT, and a custom **"budget mobile"** template tuned for Snapdragon 665 / Helio G85
- Per-layer estimated parameter count + relative inference cost — directly useful for on-device deployment decisions, something even PyTorch doesn't surface in a friendly UI

### 5.2 Animated Visualization ("wow factor")
Two modes:
- **Flow mode** — watch a real input tensor animate through layers, morphing shape visually at each step
- **Neuron mode** — zoom into a Dense layer, see actual node-connection diagrams with weights "lighting up" (3Blue1Brown-style)

### 5.3 Live Inspection Layer (the actual differentiator)
This is what no existing tool combines — Netron shows structure but never runs anything; TensorBoard shows training metrics, not architecture; CNN Explainer is fixed, not buildable.

- **ONNX Runtime Web** — load a real `.onnx` model, run actual inference entirely in-browser (WASM), no server/GPU/Colab required, model never leaves the browser (mirrors how Netron reads files locally with zero network calls)
- **Grad-CAM** — heatmap of which pixels drove a decision (single forward pass, no training needed)
- **Activation maps** — visualize real intermediate layer outputs for a specific input image
- **Feature visualization** — show what pattern/filter each neuron responds to

Combined, these let you validate an architectural idea against a real model before spending any GPU hours — directly solving the Colab-limit / long-feedback-loop pain point.

### 5.4 Mobile-Aware Validation
- Flag dynamic (`?`) dimensions in red with an explicit warning: *dynamic shape detected — NPU/DSP acceleration will be lost, falls back to CPU (~5–10× slower)*. This is a real, mobile-specific insight most tools don't surface.

### 5.5 Graph Canvas & Layer Registry
Named architectures (U-Net, MobileNet, EfficientNet) don't differ by shape math alone — shape math only answers "what size comes out." What actually distinguishes them is one of three things, none of which a linear layer-stack can express:

- **Topology** (U-Net) — an encoder/decoder graph with skip connections reaching across to matching-resolution layers. Requires the builder to support arbitrary **DAGs**, not just a linear list.
- **Layer-type substitution** (MobileNet) — e.g. depthwise separable conv in place of standard Conv2D: identical output shape, very different param/FLOP cost.
- **Scaling strategy** (EfficientNet) — a meta-rule (compound scaling) for growing depth/width/resolution together, applied on top of a building block (MBConv + SE).

Design implications:
- The builder canvas must be a node-link graph (drag, connect, branch) — a straight stack can't represent U-Net.
- The **layer registry** (type → shape function) gets a second column: **cost estimate** (param count / relative FLOPs). This is invisible to the shape check but is exactly what separates a MobileNet block from a vanilla one — and feeds directly into the §5.4 hardware-estimate work.
- Shape-function families identified so far: Dense (replaces last dim only), Conv2D/Pool (`floor((W − k + 2p)/s) + 1` per spatial axis), Flatten (product of all non-batch dims), Activation/BatchNorm (identity — shape-preserving), Reshape (total element count must match), Add/Residual (strict shape equality between branches), Concatenate (match all axes except the joined one).
- **Block templates:** prebuilt droppable subgraphs ("MBConv block," "U-Net down-block") — most named architectures are a known block, repeated and scaled with a known topology, not something built neuron-by-neuron. This is a much smaller lift than a full freeform builder and the realistic path to making §5.1 presets buildable rather than just loadable.

### 5.6 Validation Funnel (rapid hypothesis-rejection)
Reframed goal: not "assemble known architectures faster" — **kill bad/unproven ideas cheaply before they cost real GPU hours.** Four tiers, each escalating to the next only if the cheaper one passes:

1. **Shape check** (built, §2/§3) — instant, static, structural validity only.
2. **Smoke-test forward pass** — one batch of random dummy data through the real graph, in-browser. Catches what shape-math can't: NaN/Inf, broadcasting bugs in a custom layer. Still free, still seconds.
3. **"Does it learn at all" sanity check** — a handful of real training steps with real gradients on a tiny synthetic batch, in-browser via TensorFlow.js (real autodiff, WebGL/WebGPU backend). Answers "is this structurally capable of learning at all" — loss moving, gradients exploding/vanishing/frozen. Limited to small/moderate custom blocks, not full-size networks (a full EfficientNet-B7 can't sanity-train in a browser tab) — that limitation is acceptable since the goal is rejecting bad ideas cheaply, not full training.
4. **Real training on real data** — only for ideas that survive 1–3. Handled via the Kaggle relay, §6.

---

## 6. Heavy Features — Static vs Relay Split

Ten additional features were brainstormed beyond the core curriculum/platform roadmap above. They sort cleanly by where the work can actually happen — and whether that work needs real framework libraries (PyTorch/TF/onnx) running somewhere, not just shape math.

### 6.1 Static (browser-only, free, instant — no relay needed)

**File-size note:** these tasks are size-unbounded in principle, not just "fine for small files." Any ONNX model over 2GB is already forced by the format itself to split into a small graph file + separate external weight-data files (§Tech Direction / earlier size discussion) — so structural tasks below only ever need to read that small graph file, never the multi-GB weight blobs. This holds as long as the model is relay-hosted (a Kaggle/HF link TensorScope can read from directly) rather than one raw local upload, which hands the whole file to the browser in memory immediately regardless of what's actually needed from it.

- Plain-English model review/summary before running or exporting anything
- **"Trained-ness" check** (replaces an earlier, broader "training process/data overview" idea — narrowed because exact epoch/step count is only recoverable when it exists in the file): a full checkpoint with optimizer state exposes the exact epoch/step directly, no computation needed. An exported inference-only weight file (`.onnx`/`.tflite`/`.ncnn`, and often `.pth`) never has that number — only a heuristic confidence signal is possible, by comparing weight distributions against known fresh-init patterns. BatchNorm `running_mean`/`running_var` (0/1 at init, untouched if never trained) is the cleanest practical tell.
- Layer-swap tradeoff estimates (lookup-table based — known published accuracy/FLOPs for common blocks, explicitly labeled "estimated, not measured")
- Mobile hardware compute/throttle estimates (lookup-table based against known chipset benchmarks — true thermal throttling can't be simulated, only approximated)
- Structural model comparison (shape/architecture diffing between two models)
- **Hugging Face model loading** — for models with an existing ONNX export on the Hub, run fully client-side via `transformers.js` (WASM/WebGPU). Reading a model card/config from the HF Hub API is also free and static.
- Graph-canvas model building + Tiers 1–3 of the validation funnel (§5.6)

### 6.2 Relay → Kaggle kernel (CPU or GPU is a config flag, not a different code path)

**Gating note:** there is no single global file-size cutoff between "browser" and "relay" — gating is by *task feasibility*, not size. The same model file can pass a static review instantly (§6.1) while still needing the relay for an actual-computation task below. A computation task only needs to escalate when it doesn't fit ONNX Runtime Web's standard load-it-all-upfront approach (bound by the ~4GB hard ceiling noted in §7, with realistically much lower safe margins on mobile) — ideally attempted in-browser first, falling back to relay automatically on an OOM/crash rather than relying on a pre-set threshold guess.

- Format conversion (`.pth` / TF / ONNX / TFLite / NCNN) — needs real framework libraries actually installed
- Pruning & quantization — needs real tensor math on real weights
- Tier 4 full training (§5.6), and full weight-value comparison on large models
- Hugging Face models with no existing ONNX export (need conversion or a real Python `transformers` runtime)

### 6.3 "Bring Your Own Kaggle" Relay Architecture
- User supplies their own Kaggle username + API key, stored client-side only — TensorScope never holds it.
- A thin, stateless relay forwards authenticated calls to Kaggle's API (likely required — Kaggle's REST API is built for its CLI tool and probably isn't CORS-enabled for arbitrary browser origins; **unconfirmed**, needs direct hands-on testing before committing to this architecture).
- Flow: TensorScope auto-generates a notebook from a parameterized template → `kernels push` → poll `kernels status` silently in TensorScope's own UI → `kernels output` on completion → pull results back in. User never has to visit Kaggle's site or click "Run" themselves.
- **Training notebook structure** — this can't be the usual "push a notebook and pray" exploratory style, since nothing is unattended:
  - Mandatory `kernel-metadata.json` (GPU/internet toggles, dataset sources — custom training data must already exist as a Kaggle dataset *before* it can be referenced)
  - One deterministic entry point (script or notebook run straight top-to-bottom)
  - A structured status-manifest file written at the end (e.g. `{"status": "success", "epoch": 12}`) so polling code has something machine-readable to parse, instead of scraping notebook output
  - Checkpointing — Kaggle sessions cap around 9 hours; a long run getting cut off shouldn't waste the whole quota-burn
- **Output handling:** outputs attached to a saved kernel version persist until deleted or a storage cap is hit — they don't auto-evaporate on a timer. Safe pattern regardless: pull the output the moment polling shows "complete" and hand it to the user's own storage, rather than treating Kaggle as the only copy.

### 6.4 On-Device Benchmark Builder ("one-click" APK/IPA)
Goal: verify the §6.1 hardware-estimate lookup tables against ground truth on the user's actual phone — close the loop between "estimated" and "measured."

**Android — genuinely one-click, feasible:**
- Pipeline: TensorScope packages the (already mobile-format-converted, §6.2) model with a small benchmark-harness app (loads via ONNX Runtime Mobile/NCNN, runs N inference passes, logs latency + actual backend used) → builds via a CI runner (e.g. GitHub Actions) → returns a downloadable, debug-signed APK. No app-store gatekeeping required — Android allows direct sideload install.
- "One-click" means *no manual steps*, not *instant* — a real CI build still takes a couple of minutes; surface that wait in the UI rather than implying it's immediate.
- Same architectural pattern as the Kaggle relay (§6.3): a "bring your own GitHub" build, running on infrastructure TensorScope doesn't own or pay for.
- High-value payoff: Android exposes a real, queryable on-device thermal status API, and mobile inference runtimes report which backend actually executed (CPU/GPU/NNAPI/NPU) — not a guess. This makes the benchmark APK the **ground-truth verification step** for the §6.1 estimates, on real hardware.

**iOS — structurally blocked from being one-click:**
- Not an engineering limitation — it's Apple's distribution model. Every binary installed on a real device must be signed against a specific Apple Developer account/provisioning profile; there is no sideload-equivalent to Android's open install.
- Closest honest option: a "bring your own Apple Developer account" relay (same shape as §6.3), which still requires the user to already hold a paid ($99/yr) developer account and complete at least one manual trust/install step on-device. Not zero-click under any design.
- Realistic alternative: point iOS users at Apple's own Xcode/Instruments profiling tools rather than rebuilding a worse version of them.

**Dependency:** this feature only makes sense once a model is already in a mobile-deployable format (TFLite/NCNN/ONNX Mobile) — it sits *after* the §6.2 conversion relay in the pipeline, not standalone.

---

## 7. Tech Direction

- **Form factor:** Web app (chosen over VS Code extension or Python library) — maximizes reach, no install, works for non-coders too
- **Inference runtime:** ONNX Runtime Web (WASM) for models in the tens-of-MB range (fits comfortably for mobile-target nets); WebGPU as a future option only if larger models ever become relevant — not a near-term need given the target model sizes
- **Current prototype stack:** React (Claude artifact), inline styles, no external state libs

---

## 8. Non-Goals (for now)

- Not a *self-hosted* training platform — TensorScope never runs training on infrastructure it owns. Lightweight in-browser sanity-training (Tier 3, §5.6) and full training via the user's own Kaggle account (§6.3) are both in scope; a TensorScope-owned GPU backend is not.
- Not a Netron replacement — Netron's static structure view is a solved problem; TensorScope's value is interactive shape debugging + (future) live inference inspection together
- Not generating framework-specific code (PyTorch/TF) — stays framework-agnostic by design

---

## 9. Open Threads / Pending Decisions

- Exact rank/structure for how LoRA-style swappable heads might one day be visualized in TensorScope (related to, but distinct from, the separate MobileEditNet training work — keep these two projects' specs separate per current chat-splitting decision)
- Whether preset templates should be hand-authored or auto-parsed from real ONNX files
- How much of the XAI layer (§5.3) requires waiting on Stage 6 (forward pass) vs. could be prototyped earlier as a standalone demo
- Whether Kaggle's API permits direct browser-origin calls (CORS) — unconfirmed; determines whether the §6.3 relay can be skipped entirely or is mandatory
- Whether Kaggle GPU quota is queryable via the API at all, or only inferable by catching the over-quota error at request time
- Exact storage-cap specifics for kernel outputs (unconfirmed — affects how aggressively TensorScope needs to pull results off Kaggle)
- Whether HF Hub model cards/configs (§6.1) should be fetched live per-request or cached
- Whether requiring a paid Apple Developer account for the §6.4 iOS path is an acceptable bar, or whether iOS should be dropped from that feature entirely in favor of just pointing to Xcode/Instruments
- Whether a hand-built **layer-by-layer streaming executor** is worth building as a future stretch goal: reading external-data weight offsets directly and running one layer at a time (bypassing ONNX Runtime Web's standard session API, which always loads everything upfront), bounding peak memory by the single largest layer rather than the whole model — the same idea llama.cpp uses for oversized models. Not v1 scope, but the most promising lever for pushing the in-browser live-run ceiling past today's hard limit. Separately, WebGL/WebGPU also caps the size of any *single* tensor/texture, which could bite an unusually large individual layer (e.g. a giant embedding table) even before the whole-model ceiling is reached.

---

*Last updated: reflects the static-vs-relay brainstorm (graph canvas, validation funnel, "bring your own Kaggle" relay) — mid-curriculum (backpropagation), before the platform/XAI layer has been built.*
