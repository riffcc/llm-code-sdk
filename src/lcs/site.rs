use crate::lcs::hugo::{ensure_hextra_theme, ensure_site_dirs, write_text};
use anyhow::Result;
use std::path::Path;

const LCS_HUGO_CONFIG: &str = r#"baseURL: "https://lcs.riff.cc/"
languageCode: "en-gb"
title: "LCS"
theme: "hextra"
enableGitInfo: false

params:
  description: "LCS is a live codebase cognition system for Palace."
  navbar:
    displayTitle: true
    displayLogo: false
    width: wide
  page:
    width: full
  footer:
    displayPoweredBy: false

markup:
  goldmark:
    renderer:
      unsafe: true

menu:
  main:
    - name: Home
      pageRef: /
      weight: 1
    - name: Docs
      pageRef: /docs
      weight: 2
    - name: Jetpack
      pageRef: /docs/dogfooding/jetpack
      weight: 3
    - name: IPFS HA
      pageRef: /docs/dogfooding/ipfs-ha
      weight: 4
"#;

const LCS_HOME: &str = r#"---
title: LCS
---

# LCS

LCS is a live codebase cognition system embedded in `llm-code-sdk`.

It grows out of SmartRead's original five-layer code analysis system, then pushes far past it into imperative code intelligence, JIT documentation understanding, graph-backed inference, and live documentation bases kept hot by an `lcs` daemon.

{{< cards >}}
  {{< card link="/docs/introduction/what-is-lcs" title="What It Is" subtitle="The core thesis and product shape." >}}
  {{< card link="/docs/introduction/five-layer-foundation" title="Five-Layer Core" subtitle="The SmartRead foundation LCS is built on." >}}
  {{< card link="/docs/architecture/semantic-graph" title="Semantic Graph" subtitle="The substrate under every page and glass." >}}
  {{< card link="/docs/architecture/extensions" title="Extensions" subtitle="The wider surface around the original five layers." >}}
  {{< card link="/docs/architecture/glasses" title="Lens Inference" subtitle="Automatic audience bias with optional override." >}}
  {{< card link="/docs/architecture/correction-wave" title="Correction Wave" subtitle="How accepted edits become propagation, not drift." >}}
  {{< card link="/docs/dogfooding/jetpack" title="Jetpack Test Case" subtitle="The first real repository target." >}}
  {{< card link="/docs/dogfooding/ipfs-ha" title="IPFS HA Dogfood" subtitle="Cross-repo add-path stitching with proof anchors." >}}
{{< /cards >}}
"#;

const DOCS_INDEX: &str = r#"---
title: Docs
cascade:
  type: docs
---

LCS turns repository structure into a living documentation surface.
"#;

const INTRO_INDEX: &str = r#"---
title: Introduction
---

Foundational pages that define what LCS is and why it exists.
"#;

const ARCH_INDEX: &str = r#"---
title: Architecture
---

Core system ideas: semantic graph, lens inference, and correction propagation.
"#;

const DOGFOOD_INDEX: &str = r#"---
title: Dogfooding
---

Real repositories used as proof targets for the LCS publication pipeline.
"#;

const ROADMAP_INDEX: &str = r#"---
title: Roadmap
---

Execution slices and workstream structure for shipping LCS.
"#;

const WHAT_IS_LCS: &str = r#"---
title: What Is LCS
weight: 1
---

LCS is short for `llm-code-sdk`, but as a product it refers to the live code understanding system being built inside that crate.

The basic move is simple:

1. Parse code into semantic objects.
2. Trace dependencies and paths through those objects.
3. Answer code and docs questions through imperative tools.
4. Keep live documentation bases hot from the same substrate.
5. Let corrections flow back into the graph and regenerate downstream pages.

The canonical verbs are `smart_read`, `smart_write`, `ask_docs`, `ask_code`, and `mr_search`.

The split between `smart_read` and `ask_code` matters:

- `smart_read` is the fast grounded structural answer path.
- `ask_code` is the real query surface: agentic, investigative, and willing to roam through the codebase with tools before it answers.

This is why LCS is not just a docs generator, and not even just a semantic wiki engine. The Hugo surface is one manifestation of a much larger system.

The center of gravity is codebase cognition:

- SmartRead's multi-layer structural understanding
- graph-backed retrieval and snapshotting
- `ask_code` as an evidence-grounded query surface in the DeepWiki lineage, but less rigid and more tool-native
- JIT indexing for docs and repository context
- model-assisted inference over code and docs
- live operator-facing and reader-facing surfaces that update as the system learns

The publishing surface matters, but it is downstream of the analysis system rather than the other way around.
"#;

const FIVE_LAYER_FOUNDATION: &str = r#"---
title: Five-Layer Foundation
weight: 2
---

LCS starts from SmartRead's original five-layer analysis system.

That foundation was loosely inspired by compiler-theory-driven approaches such as Continuous Claude's structural slicing ideas, but adapted into Palace and `llm-code-sdk` as a practical tool for understanding real repositories quickly.

The five base layers are:

1. AST: declaration boundaries, symbols, imports, structure
2. Call graph: which functions and methods invoke which others
3. CFG: control-flow shape inside functions and handlers
4. DFG: data flow, definitions, uses, and dependency chains
5. PDG: combined control and data dependence for slicing

Those five layers were the original SmartRead unlock. They made it possible to inspect a codebase at multiple structural wavelengths instead of only reading raw files.

But LCS is what happens when that idea stops being a single read tool and becomes a full system.
"#;

const SEMANTIC_GRAPH: &str = r#"---
title: Semantic Graph
weight: 1
---

The semantic graph is the heart of LCS.

It models files, modules, symbols, tests, benchmarks, routes, configuration keys, pages, and corrections as first-class nodes with typed edges like `defines`, `imports`, `calls`, `reads`, `writes`, `documents`, and `invalidates`.

The current SmartRead layers are the beginning of that substrate, not the end of it. AST, call graph, CFG, DFG, and PDG become projections over one graph rather than separate silos.
"#;

const EXTENSIONS: &str = r#"---
title: Extensions
weight: 2
---

LCS is the extension field around the original five layers.

The five-layer system is the seed. The real product is the much wider surface that grows around it:

- imperative verbs like `smart_read`, `smart_write`, `ask_code`, `ask_docs`, and `mr_search`
- `smart_read` for fast grounded structural reads, `ask_code` for actual codebase investigation
- query-style answers that synthesize from SmartRead, Lean theory graphs, stitched repo snapshots, and live docs evidence
- JIT indexing instead of static up-front indexing ceremonies
- graph-backed snapshots that can cut through a codebase along exact paths
- lens inference so the same substrate can answer developer, API, user, or operator questions differently
- live Hugo bases that stay hot while the underlying repository evolves
- local and remote model routing for private or scaled inference
- correction propagation so edits improve downstream generated surfaces instead of rotting as one-offs
- repository daemons that keep the knowledge surface synchronized continuously

So when someone says LCS is “the wiki thing”, that is only one visible edge of the system.

The real aim is to make codebases inspectable from hundreds of angles, at multiple scales, with much lower latency and much better reuse than raw-file prompting.
"#;

const GLASSES: &str = r#"---
title: Lens Inference
weight: 3
---

Audience bias is a projection over the same codebase graph.

LCS should infer the right lens automatically for most Palace calls.

- Developer bias emphasizes internals, runtime paths, hot edges, and architecture boundaries.
- API bias emphasizes contracts, inputs, outputs, handlers, and request flow.
- User bias emphasizes workflows, concepts, and task-oriented documentation.
- Operator bias emphasizes deployment, failure handling, configuration, and recovery.

When the caller wants a specific viewpoint, the tool call can override the inferred lens explicitly. Inside the Hugo site, the reader should be able to switch lenses directly.

The important property is consistency. The graph stays the same. The projection changes.
"#;

const CORRECTION_WAVE: &str = r#"---
title: Correction Wave
weight: 4
---

The correction wave is the feature that makes LCS living instead of static.

An edit against a rendered claim is resolved back to semantic objects, reviewed, accepted, and then propagated across every page derived from the corrected fact.

That gives LCS a route from human intervention back into machine-generated documentation without turning the site into a pile of one-off overrides.
"#;

const JETPACK_DOGFOOD: &str = r#"---
title: Jetpack Test Case
weight: 1
---

Jetpack is the first deliberate dogfood target for LCS.

It has:

- a clean Rust codebase,
- clear subsystems,
- real examples,
- obvious developer and operator audiences,
- and existing documentation debt that benefits from structured regeneration.

The generated companion site lives separately as `jetpack-docs`, but it is produced by LCS code inside `llm-code-sdk`. The same substrate should also power `ask_code`, `ask_docs`, and `mr_search` against Jetpack.
"#;

const IPFS_HA_DOGFOOD: &str = r#"---
title: IPFS HA Dogfood
weight: 2
---

The IPFS HA dogfood surface is the first stitched cross-repo view aimed at optimization work.

It is intentionally narrow:

- ingress split in nginx
- Cluster add facade
- direct Kubo add path
- shared-repo patch surface
- flatfs-mw async publish path
- Lean proof anchor for the multiwriter safety claim

This is the right first LCS move for the storage work because it makes one real hot path inspectable end to end instead of waiting for a perfect generic semantic graph before anything is useful.
"#;

const WORKSTREAMS: &str = r#"---
title: Workstreams
weight: 1
---

The initial LCS execution map spans 17 workstreams:

- Semantic Graph Core
- Source Ingestion and Parsing
- Semantic Extraction Layers
- Retrieval and Snapshot Engine
- Hugo and Hextra Publishing Pipeline
- Live Regeneration and Hot Production Reload
- Correction Propagation and Self-Healing Wiki
- Editable Wiki Authoring and Interaction Layer
- Multilingual and Translation System
- Search, Chat, and Conversational Guidance
- Synthesis and Local Model Runtime
- Lean and Formal Methods Observatory
- Provenance, Trust, and Review Workflow
- Operator Tooling and Control Plane
- Security, Auth, and Multi-Tenant Isolation
- Performance, Caching, and Reliability
- Commercial Productization and Launch
"#;

pub fn generate_lcs_site(output: &Path) -> Result<()> {
    ensure_site_dirs(output)?;
    ensure_hextra_theme(output)?;

    write_text(output, "hugo.yaml", LCS_HUGO_CONFIG)?;
    write_text(
        output,
        "layouts/_partials/scripts/search.html",
        "{{/* Search disabled for offline local builds. */}}\n",
    )?;
    write_text(output, "content/_index.md", LCS_HOME)?;
    write_text(output, "content/docs/_index.md", DOCS_INDEX)?;
    write_text(output, "content/docs/introduction/_index.md", INTRO_INDEX)?;
    write_text(output, "content/docs/architecture/_index.md", ARCH_INDEX)?;
    write_text(output, "content/docs/dogfooding/_index.md", DOGFOOD_INDEX)?;
    write_text(output, "content/docs/roadmap/_index.md", ROADMAP_INDEX)?;
    write_text(
        output,
        "content/docs/introduction/what-is-lcs.md",
        WHAT_IS_LCS,
    )?;
    write_text(
        output,
        "content/docs/introduction/five-layer-foundation.md",
        FIVE_LAYER_FOUNDATION,
    )?;
    write_text(
        output,
        "content/docs/architecture/semantic-graph.md",
        SEMANTIC_GRAPH,
    )?;
    write_text(
        output,
        "content/docs/architecture/extensions.md",
        EXTENSIONS,
    )?;
    write_text(output, "content/docs/architecture/glasses.md", GLASSES)?;
    write_text(
        output,
        "content/docs/architecture/correction-wave.md",
        CORRECTION_WAVE,
    )?;
    write_text(
        output,
        "content/docs/dogfooding/jetpack.md",
        JETPACK_DOGFOOD,
    )?;
    write_text(
        output,
        "content/docs/dogfooding/ipfs-ha.md",
        IPFS_HA_DOGFOOD,
    )?;
    write_text(output, "content/docs/roadmap/workstreams.md", WORKSTREAMS)?;
    write_text(
        output,
        "static/css/custom.css",
        ":root { --lcs-accent: #183a5a; }\n",
    )?;

    Ok(())
}
