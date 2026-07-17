# NeoDiffusion GUI — Design & Implementation Guide

Status: draft v3, final — all open design questions resolved.
Owner: André. Companion doc to `CLAUDE.md` / `Plans/handoff-post-M5.md` — read those for engine context first.

## 1. Concept: "Latent Canvas"

The whole GUI should visually behave like the thing it renders: a field that resolves from noise into signal.

- Chat is not a list of finished bubbles appearing instantly — each assistant turn is a canvas that visibly denoises from masked glyphs into final text, block by block.
- The visual language borrows directly from the block-causal / block-diffusion mechanics already in the engine: a block is a physical region on screen, not just a text range. Finished blocks compress into normal prose; the active block is the only place "work" is visibly happening; future blocks are inert placeholders.
- Translucency is used with intent, not decoration: materials get **less transparent as content resolves**. A masked token sits behind a frosted, low-opacity chip; a committed token sits on fully opaque background. This makes "resolution = clarity" literal, not just metaphorical.
- This concept must hold in both Pleasant and Debug mode — Debug mode adds data density (confidence heat, edit flags, step counters), it does not change the underlying metaphor.

## 2. Typography

- **Lexend** for all UI chrome, chat prose, labels, menus. Lexend was designed for reading proficiency — appropriate given this is a research tool meant to be read closely, not skimmed.
- **Illinois Mono** for anything token-shaped: code blocks, the flag panel's raw parameter values, `/`-command input, StepTrace debug readouts, model identifiers. This gives an immediate visual cue: "if it's mono, it's literal/machine-truth; if it's Lexend, it's prose/UI."
- Hierarchy: establish 4 sizes only (large title for turn headers, body for prose, caption for metadata/timestamps, mono-caption for debug numerics). Avoid ad hoc font sizes — resist the temptation to make "important" flags bigger; use weight and color, not size, for emphasis inside a given tier.
- Register both fonts as custom fonts in the app bundle (`ATSApplicationFontsPath` in the bundle Resources on macOS, or embed via `Font.custom` after registering the `.otf`/`.ttf` files) — confirm Illinois Mono's license permits bundling before shipping outside your own machine.

## 3. Visual system — translucency & materials (SwiftUI/AppKit-native)

- Base window: `.background(.ultraThinMaterial)` or an `NSVisualEffectView`-backed material, respecting light/dark and the "Reduce Transparency" accessibility setting (SwiftUI materials degrade gracefully automatically — do not hardcode opacity as a workaround).
- Masked-token chip: custom `ShapeStyle` combining `.thinMaterial` + low-opacity tint tied to the model's mask color; opacity animates toward 1.0 as `transferred` flips true in the `StepTrace` stream.
- Active-block container: subtle `.regularMaterial` panel distinct from the finished-text background, with a soft animated border (avoid heavy glow/neon — keep it restrained, matching a research tool rather than a marketing demo).
- Respect macOS conventions: sidebar (flag/model panel) should feel like a native `NavigationSplitView` sidebar with vibrancy, not a custom-drawn overlay.
- Motion: use `.animation(.easeOut, value:)` scoped narrowly per-token, not global re-renders — with block sizes in the tens of tokens this must stay cheap; profile before shipping.

## 4. Phases

Scope is wide enough that it should not be built as one PR. Phases are ordered by dependency, not by "nice to have."

### Phase 0 — Foundations (no visuals yet)

- Define `GenerationConfig` (immutable value type, `Codable`) mirroring the `effective*` fields on `DiffusionEngine.Metrics` (nBuf, tauAdd, tauSemi, speculationK, dynamicTauAlpha, jotEnabled/K/Threshold/Faithful, iceEnabled/Tau/Nt/ThinkingLength, creditDecodingEnabled/Alpha/Beta/Gamma, moeCapacityRatio, eosEarlyExit).
- **Flag configuration source of truth**: a `Settings.swift` file holds the canonical schema — one static, `Codable` struct with a property per flag, grouped to mirror `Metrics.effective*`, plus a sensible default value for each. This is your compile-time toggle surface: enable/disable or re-default any flag by editing `Settings.swift` and rebuilding, matching how you already work (Xcode build cycle, not a live-reload workflow).
  - Layer an optional `config.json` (read once at launch from the app's Application Support directory) on top of `Settings.swift`'s defaults, so you can override a subset of flags without recompiling when useful — but `Settings.swift` stays the schema owner; generate/validate `config.json`'s shape from it rather than maintaining two independent definitions.
  - Each flag entry also carries a `stability: .stable | .experimental` case: the flag panel's "Stable" vs "Experimental / pending validation" tiering (Phase 2) reads this marker directly instead of a hand-maintained separate list, so a flag's tier can never silently drift out of sync with its actual definition.
- Define `ConnectionTarget`: local in-process engine vs. remote HTTP endpoint (host:port over Tailscale). Both must conform to one `InferenceBackend` protocol so the rest of the app never branches on transport.
- Wire a minimal `NetworkBackend` against the existing M7 OpenAI-compatible server. No TLS assumptions — Tailscale traffic is already encrypted at the network layer, so plain HTTP inside the tailnet is acceptable; do not over-engineer auth here.
- **Device discovery (MagicDNS-style)**: query Tailscale's local API (`tailscale status --json`, or the daemon's loopback endpoint) to list peer devices on the tailnet with their MagicDNS names, and surface them as a picker instead of a free-text hostname field. Practical notes:
  - Requires the Tailscale CLI/daemon present on the machine running the GUI — if it's missing or the query fails, fall back gracefully to manual hostname entry rather than blocking the feature.
  - The picker should show device name + a live reachability ping result (small colored dot), so you don't pick a peer that's asleep or has the server not running.
  - Cache the last successful connection choice in `AppStorage` so reopening the app doesn't require re-discovery every time.
- Basic model registry: static list of supported models (see §7) with an enum/struct tagging capability flags (e.g. supportsBlockDiffusion, supportsSpeculation) so the UI can hide irrelevant flags per model without hardcoding per-model UI branches later.

### Phase 1 — Chat shell, Pleasant mode only

- Multi-turn chat list (`NavigationSplitView`: conversation list sidebar + main chat column), materials per §3.
- Send / delete / undo / copy actions via SF Symbols icon buttons (`paperplane.fill`, `trash`, `arrow.uturn.backward`, `doc.on.doc`) with Lexend tooltips, not permanent text labels — keep the input bar visually quiet.
- Render assistant turns using the "Latent Canvas" concept: masked chip → resolved text, driven by `StepTrace` from a **local** generation run only (network streaming deferred to Phase 4).
- Persistence groundwork: even though the flag panel isn't built yet, wire the conversation store now (see Phase 2's persistence note) so chat history survives from day one rather than being retrofitted later.

### Phase 2 — Flags, slash commands, persistence

- Flag panel as a collapsible inspector/sheet, editable only while idle (disabled entirely while `isGenerating == true`, per your earlier decision). Sourced from `Settings.swift`/`config.json` (Phase 0) and tiered Stable vs Experimental using each flag's `stability` marker.
- `/`-command parser on the chat input:
  - `/q` — quality-mode preset (conservative tau / `.strict` decoding).
  - `/f` — fast-mode preset, and also the namespace for optimization flags with explicit params, e.g. `/f nbuf=2 tauAdd=0.85 speculationK=4` — `/f` alone applies a fast-mode default bundle, while `/f <flag>=<value> ...` overrides individual params on top of that bundle in one line. Keep the parser grammar as `key=value`, space-separated, matching the property names in `GenerationConfig`.
  - `/reset` — revert `draftConfig` to `Settings.swift` defaults.
  - `/model <name>` — switch active model from the registry.
  - All commands mutate the same `draftConfig` the menu edits — one source of truth, two entry points.
- Autocomplete popover when the user types `/` — list available commands with one-line descriptions; for `/f`, show available param names as you type past the space, so you don't need to memorize exact flag spellings. Illinois Mono for the command/param tokens, Lexend for descriptions.
- Snapshot config at "Run" time into an immutable value passed to the backend — panel and slash commands only ever touch the mutable draft.
- **Persistence (per-machine only, no cloud sync)**: use libSQL via the official `tursodatabase/libsql-swift` Swift package for conversation storage.
  - Local-only: `Database(path: "conversations.db")` — a pure on-device SQLite-compatible file, stored in the app's Application Support directory, zero network dependency and no auth tokens to manage.
  - Explicitly no Turso Cloud sync — each Mac keeps its own independent history. If cross-machine sync is ever wanted later, the same package supports an embedded-replica sync mode without a storage-layer rewrite, but this is out of scope for now.
  - Schema: `conversations(id, title, createdAt)`, `turns(id, conversationId, role, text, config_json, trace_json_ref, createdAt)` — store the snapshotted `GenerationConfig` alongside each assistant turn so any past turn can be inspected or regenerated with its exact settings later.
  - Note the package is in technical preview per its own README — pin an exact version in `Package.swift` and re-test after upgrades rather than tracking latest automatically.

### Phase 3 — Debug mode

- Mode toggle (segmented control or menu item): Pleasant vs Debug — swaps what `TraceRenderer` draws, not the underlying data pipeline.
- Confidence heat: map `StepTrace.confidence[i]` to opacity/hue on each token chip.
- Edited-token indicator: subtle border flash when `StepTrace.edited[i]` is true (committed once, then revised) — diagnostically the most interesting signal given known quantization drift.
- Step counter badge per block (from `stepsPerBlock` / `Metrics.postStepsPerBlock`) and a small live sparkline of mean transfer confidence per step (`meanTransferConfidencePerStep`).
- Per-run export: serialize `(GenerationConfig, [StepTrace], Metrics)` to one JSON file, so debug sessions slot into your existing `Plans/*-logbook.md` workflow. Also write a reference/pointer to this file into the `turns` table (Phase 2) so a past chat turn can be traced back to its full debug artifact.

### Phase 4 — Multi-model & remote polish

- Model picker wired to the registry from Phase 0: LLaDA2.1-mini and Sumi first (already implemented engine-side); LLaDA2.1-flash, fast-dLLM (both sizes), and the Zigeng/dmax collection (https://huggingface.co/collections/Zigeng/dmax-models) added as they land engine-side — UI should only need registry entries, not structural changes, if Phase 0's capability-flag design holds.
- Full remote streaming parity: `StepTrace` events forwarded over the network connection (SSE or WebSocket extension to the M7 server) so Debug mode works identically local or across Tailscale.
- Connection status affordance (small badge: local / connected via Tailscale to `<device-name>` / disconnected), reusing the MagicDNS picker from Phase 0 and consistent with the existing hardware-status-badge pattern already in `ContentView.swift`.

## 5. Icon set (SF Symbols, initial pass)

| Action | Symbol |
|---|---|
| Send | paperplane.fill |
| Delete (message/turn) | trash |
| Undo (last edit/turn) | arrow.uturn.backward |
| Copy | doc.on.doc |
| Regenerate | arrow.clockwise |
| Flags/settings | slider.horizontal.3 |
| Debug mode toggle | eye / eye.slash or waveform |
| Connection status | antenna.radiowaves.left.and.right |
| Model picker | cpu |

Use `Label` with `systemImage` throughout so VoiceOver labels come for free — avoid bare `Image(systemName:)` buttons without an accessibility label.

## 6. Networking (Tailscale) notes

- Treat "remote" purely as a different `InferenceBackend` implementation hitting `http://<tailscale-device>:<port>/v1/...` (the existing M7 OpenAI-compatible surface) — no VPN-specific code needed, Tailscale just makes the hostname routable.
- Discovery via the local Tailscale API/CLI (§4 Phase 0) removes the need to remember or type hostnames; store the last-picked device in `AppStorage` as a fallback default.
- Add a lightweight reachability/health-check ping before enabling "Run" against a remote target, so failures surface as a connection badge state, not a silent hang.

## 7. Model roadmap awareness

Current engine support: LLaDA2.1-mini, Sumi. Planned: LLaDA2.1-flash, fast-dLLM (both sizes), and the DMax model family. The registry/capability-flag design in Phase 0 exists specifically so adding these later is a data change, not a UI rewrite — worth validating this holds by trying it once with a second real model early (e.g. Sumi) rather than only at the end.

## Resolved design decisions

1. Network discovery: MagicDNS-style device picker via Tailscale's local API, with manual-hostname fallback.
2. Slash commands: `/q` (quality), `/f` (fast preset, plus `key=value` param overrides for optimization flags), `/reset`, `/model <name>` — grows organically from here.
3. Persistence: yes, via libSQL (`tursodatabase/libsql-swift`), per-machine local file only — no Turso Cloud sync.
4. Flag config: `Settings.swift` as compile-time schema/defaults, optional `config.json` override at launch; each flag carries a `stability` marker driving the Stable/Experimental UI tiering automatically.

All open design questions are resolved; the plan is ready to move into Phase 0 implementation.
