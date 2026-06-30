# TensorScope — Build Plan & Implementation Guide

**Companion to:** `tensorscope_feature_spec.md` (the *what/why*). This document is the *how/when* — a dependency-ordered build sequence with concrete tasks, so a developer can pick this up and know exactly what to build next without re-deriving the architecture.

---

## How to Use This Document

Each milestone is something **demoable on its own** before moving to the next. Don't start a milestone until the previous one's "Done when" line is true — this isn't bureaucracy, it's the project's own philosophy (§5.6 of the spec: validate cheap before escalating expensive) applied to the build process itself. Build the cheapest, static, highest-confidence pieces first; only reach for the relay/server pieces once there's a real product underneath them worth connecting to.

---

## Team & Skillset Needed

If this is one person wearing every hat, that's fine — the order below is also a learning curriculum, not just a build order. If it's a small team, these are the natural split points:

| Role | Where they work | Core skills |
|---|---|---|
| Frontend/Logic dev | M1–M9, M14 | JS/TS, React, basic linear algebra |
| Infra/Backend dev | M10–M13 | Node or Python, serverless functions, CI (GitHub Actions), REST APIs |
| ML-adjacent dev | M5–M8, M11–M12 | ONNX format, TensorFlow.js, quantization basics |

---

## Tech Stack at a Glance

- **Frontend:** React + TypeScript, hosted as a static site (Vercel/Netlify/GitHub Pages)
- **In-browser inference:** ONNX Runtime Web (WASM, WebGPU where available)
- **In-browser training (Tier 3 sanity-check only):** TensorFlow.js
- **Relay:** one small stateless serverless function (Cloudflare Worker or Vercel serverless function) — forwards the user's own Kaggle credentials to Kaggle's API, never stores them
- **Heavy jobs:** Kaggle Kernels (user's own account/quota), triggered via the relay
- **Android benchmark builds:** GitHub Actions, debug-signed APK
- **iOS:** out of scope for one-click (see spec §6.4) — not part of this build plan's critical path

---

## Milestone 0 — Project Scaffold

**Goal:** a repo that runs, deploys, and has a CI pipeline, before any real feature exists.
**Build:**
- React + TypeScript app, basic routing, deployed to a static host on every push
- Linting/formatting/test runner wired into CI from day one
- A `layers/` module folder and a `kaggle-relay/` (or similar) folder reserved up front, even empty — this keeps the eventual relay boundary clean rather than something bolted on later
**Skills:** basic frontend tooling
**Done when:** an empty page deploys automatically on push to `main`.

## Milestone 1 — Shape Engine Core (no UI yet)

**Goal:** the actual brain of the tool, as a pure, framework-free, fully unit-tested module. Everything else is built on top of this.
**Build:**
- A **layer registry**: a map of layer type → shape function. Implement the families from spec §5.5: Dense, Conv2D, Pool, Flatten, Activation/BatchNorm (identity), Reshape, Add/Residual, Concatenate.
- Each entry takes input shape(s), returns either an output shape or a structured error (not a thrown exception — the UI needs to *display* the mismatch, not catch a crash).
- Represent a model as a simple linear list first (a DAG comes in M4) — array of layer configs, each referencing the previous layer's output shape.
**Skills:** plain JS/TS, the shape math from spec §5.5
**Done when:** you can feed a hand-written array of layer configs into a function and get back either a valid final shape or a specific, correct error — fully covered by unit tests, zero UI.

## Milestone 2 — MVP Linear Builder UI

**Goal:** the first thing a user actually sees and touches.
**Build:**
- A simple list-based builder: add a layer, pick its type, set its params, see the running shape after each one
- Green/red indicator per layer, using M1's engine directly
- This is intentionally *not* a graph canvas yet — that's M4. Don't build branching support here; resist the pull to do it now.
**Done when:** a non-technical user can stack Dense/Conv/Flatten layers and immediately see where a shape mismatch happens, with no relay, no backend, no auth.

## Milestone 3 — Plain-English Review

**Goal:** spec §6.1's core static feature — the tool's actual differentiator.
**Build:**
- For each layer in the registry, write a natural-language template describing what it does and why the shape changed the way it did (this content doubles as the explanation text from earlier design conversations — e.g. "this layer multiplies each input by its own weight and adds them up")
- A single "Review this model" view that walks the whole stack in plain English
**Done when:** pasting in a model produces a paragraph a non-expert could read and roughly understand what the network does.

## Milestone 4 — DAG Canvas Upgrade

**Goal:** support real architectures (U-Net, ResNet) — spec §5.5.
**Build:**
- Swap the linear list for a node-link graph (drag, connect, branch). Recommended: a lightweight existing graph-canvas library rather than hand-rolling drag/connect physics from scratch — this is a solved UI problem, don't re-solve it.
- Extend M1's engine to walk a DAG: `Add`/`Residual` nodes need strict shape equality across incoming edges; `Concatenate` needs axis-matching logic (already designed in §5.5)
- Add the **cost column** to the registry now (param count / relative FLOPs per node) — needed later for M8 and M14, easiest to add while you're already touching every registry entry
**Done when:** a hand-wired U-Net-shaped graph (downsample path, upsample path, skip connections) validates correctly, including a deliberately broken skip connection showing the right error.

## Milestone 5 — Smoke-Test Forward Pass (Tier 2)

**Goal:** spec §5.6 Tier 2 — catch what shape math can't.
**Build:**
- Integrate ONNX Runtime Web; run one batch of random dummy data through the graph for real
- Surface NaN/Inf or runtime errors back to the same review UI from M3
**Watch out for:** this is where the ~4GB / external-data realities from the size discussion start to matter — gate this action by *attempting it and catching failure*, not a hardcoded file-size check (per spec §6.2's gating note).
**Done when:** a model with a subtly broken custom layer (passes shape check, fails at runtime) gets caught and clearly reported.

## Milestone 6 — Sanity-Check Training (Tier 3)

**Goal:** spec §5.6 Tier 3 — "does this even learn."
**Build:**
- Integrate TensorFlow.js; run a handful of real training steps with real gradients on a tiny synthetic batch
- Report loss trend + a gradient-health flag (exploding/vanishing/frozen)
**Done when:** a known-bad architecture (e.g. no activation functions anywhere) visibly fails this check, and a normal small architecture visibly passes it, within seconds.

## Milestone 7 — Trained-ness Heuristic

**Goal:** spec §6.1 — tell a real pretrained model apart from an untouched/initialized one.
**Build:**
- Read BatchNorm `running_mean`/`running_var` where present; flag "looks untrained" if still at init defaults (0/1)
- Fallback: compare general weight-value distributions against known init patterns for layers without BN
- Label results clearly as a confidence signal, never a certainty
**Done when:** a freshly-initialized model and a genuinely trained model of the same architecture produce visibly different labels.

## Milestone 8 — Hardware Estimate Tables

**Goal:** spec §5.4/§6.1 — mobile-aware estimates without needing real hardware.
**Build:**
- A lookup table of known chipset benchmark figures (start with your own target chips — Snapdragon 665, Helio G85)
- Use M4's cost column (param count/FLOPs) to estimate compute time per chip, labeled explicitly as "estimated, not measured"
- Flag dynamic (`?`) shapes in red per spec §5.4 (NPU/DSP acceleration lost, falls back to CPU)
**Done when:** loading a model produces a labeled, honest estimate table — and a model with a dynamic dimension shows the specific NPU-fallback warning.

## Milestone 9 — Hugging Face Loader

**Goal:** spec §6.1/§6.2 — load real-world models, split correctly between static and relay.
**Build:**
- Fetch model card/config from the HF Hub API (free, static, no relay)
- If an ONNX export exists on the Hub: load and run via `transformers.js`, fully static
- If not: flag clearly that this model needs the relay (M10+), don't silently fail
**Done when:** both cases are demonstrably handled — one model runs entirely client-side, another correctly routes to "needs relay."

---

**Checkpoint:** everything above is shippable as a complete, useful, zero-backend product. M10 onward is where real infrastructure cost and complexity begin — make sure M0–M9 actually work end-to-end before investing here.

---

## Milestone 10 — Kaggle Relay Foundation

**Goal:** spec §6.3 — the "bring your own Kaggle" pipeline, built as plumbing only, no real job yet.
**Build:**
- The relay function itself: accepts the user's Kaggle username/key (entered client-side, never stored server-side) and forwards calls to Kaggle's API
- **First task, before building anything else here:** confirm directly whether Kaggle's API actually requires this relay (CORS test) or whether the browser can call it directly — this was flagged as unconfirmed in spec §9, resolve it before writing more code around the assumption
- Build the smallest possible round-trip: push a trivial "hello world" notebook, poll status, pull output — prove the whole loop before any real training logic
**Done when:** a trivial notebook can be pushed, run, and its output retrieved, entirely through TensorScope's UI, with the user never visiting Kaggle's site.

## Milestone 11 — Conversion / Pruning / Quantization Jobs

**Goal:** spec §6.2 — the first *real* relay jobs.
**Build:**
- A parameterized notebook template per job type (conversion, pruning, quantization), each ending in the structured status-manifest file from spec §6.3 (not just human-readable notebook output)
- Checkpointing logic for anything that might approach Kaggle's ~9hr session cap
**Done when:** a real `.pth` file can be converted to ONNX through the full relay round-trip, with TensorScope correctly parsing the job's success/failure from the manifest file alone.

## Milestone 12 — Full Training Jobs (Tier 4)

**Goal:** spec §5.6 Tier 4 — only for ideas that already survived Tiers 1–3.
**Build:**
- Extend the M11 template pattern to a real training loop, with the user's own dataset (must already exist as a Kaggle dataset — confirm this UX explicitly, it's a real extra step for the user, don't hide it)
- Output: trained weights pulled back immediately on completion (per spec §6.3's "don't treat Kaggle as the only copy" rule)
**Done when:** a small real model trains end-to-end via the relay and lands back in TensorScope as a usable artifact.

## Milestone 13 — On-Device Benchmark Builder (Android)

**Goal:** spec §6.4 — ground-truth check against M8's estimates.
**Build:**
- A minimal benchmark-harness Android app template (loads a model via ONNX Runtime Mobile or NCNN, runs N passes, logs latency + actual backend used + live thermal status)
- GitHub Actions workflow that injects the user's converted model (from M11) into the template and produces a downloadable debug-signed APK
**Done when:** a real APK installs on a real budget Android device and reports real numbers that can be compared directly against M8's estimate.

## Milestone 14 — Block Templates & Presets

**Goal:** spec §5.5/§5.1 — make named architectures buildable, not just loadable.
**Build:**
- A handful of prebuilt droppable subgraphs (MBConv block, U-Net down-block) using M4's DAG canvas as prefab units
**Done when:** a user can drop in a "U-Net down-block" template and get a correct, pre-validated subgraph rather than wiring it node-by-node.

## Milestone 15 — Stretch / Explicitly Deferred

Not part of the critical path — pick up only after 0–14 are solid:
- iOS benchmark path (spec §6.4 — structurally limited, low ROI relative to effort)
- Layer-by-layer streaming executor for >4GB live-run support (spec §9 — a serious standalone build)
- Full XAI/live inspection layer — Grad-CAM, animated forward-pass trace (spec §5.3)

---

## Cross-Cutting Reminder

Every milestone from M5 onward touches a real, previously-flagged uncertainty (CORS, Kaggle quota visibility, storage caps, WASM memory ceilings). Don't architect around an assumption for any of these — spend twenty minutes confirming it directly before building a whole milestone on top of a guess. That's cheaper than finding out at M12.
