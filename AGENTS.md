# AGENTS.md

This document defines how coding agents work on **manifesto**. It is a contract, not a style guide.

manifesto is the **compiler/optimizer/codegen/scheduler/IR pipeline** for the puter platform. ARCHITECTURE.md lives in the sibling repo `../puter/ARCHITECTURE.md`; the gap inventory in `../puter/GAPS.md`. The shared kernel-level coding contract lives in `../puter/AGENTS.md`. **Read those three documents before changing anything in this repo.**

This file holds the manifesto-specific addenda.

---

## 1. What manifesto is, and is not

manifesto compiles YAML manifests into a `Topology` (ARCHITECTURE.md §6) that the runtime executor dispatches against `device.Backend` (declared in `../puter/device/interface.go`).

The pipeline stages (ARCHITECTURE.md §1):

```
asset/*.yml → parse → expand → lower → unify (PortType, §4) → optimize (FusionAST, §4.3) →
codegen (LLVM / MSL / PTX / HLO, §3.2) → schedule (liveness + interval coloring + DAG, §4–§5) →
ir.Topology → executor → device.Backend
```

manifesto **is not**:

- A model zoo. There is no `manifesto/diffusion/`, `manifesto/llama/`, `manifesto/flux/` package. Model architectures live in `asset/template/model/` as YAML.
- A runtime executor for model-specific operations. The executor dispatches step kinds like `math.matmul`, `attention.sdpa`, `shape.transpose` — never `diffusion.prepare_latents` or `scheduler.timesteps` (where "scheduler" means a noise scheduler).
- A place for Go fast-paths "until the primitives exist". If a primitive is missing, add the primitive — both the `device.Backend` method (in puter) and the `template/operation/*.yml` recipe — before any Go consumer.

The diffusion-Go contamination documented in `../puter/GAPS.md §6.5` is the named example of this anti-pattern. The presence of `manifesto/diffusion/`, `manifesto/runtime/scheduler.go` (FLUX scheduler), `manifesto/runtime/executor_diffusion.go`, and `manifesto/runtime/latents.go` violates §1 of this document and is slated for removal per the sequencing in GAPS.md.

---

## 2. Closed-world atomic op set

The set of ops manifesto can emit is exactly the set declared on `device.Backend` in `../puter/device/interface.go`. There is no plugin path, no dynamic kernel registry, no string-keyed lookup at execution time.

When a YAML manifest references an op the closed-world set doesn't have, the right response is:

1. Add the method to `device.Backend` with the zero-host-sync signature.
2. Implement it across all backends (CPU scalar + AVX-512 + AVX2 + SSE2 + NEON + Metal + CUDA + XLA) with parity tests. Per `../puter/AGENTS.md §1`.
3. Add the `template/operation/<family>/<op>.yml` recipe in `asset/`.
4. Then write the manifest that consumes it.

Never the reverse. Never inline the op in Go because the primitive is missing.

---

## 3. Asset format (`kind:`)

Every YAML in `asset/` starts with `kind: <Kind>` on the first non-empty, non-comment line. The 13 remaining old-format files are listed in `../puter/GAPS.md §6.5` and need migration. New assets must use the new format.

`scripts/check_banned.sh §5` enforces this.

---

## 4. IR shape

The IR types (`ir/topology.go`, `ir/node.go`, `ir/port.go`, `ir/edge.go`) must converge on ARCHITECTURE.md §6:

```go
Topology { Nodes, Edges, Workspace, InputPorts, OutputPorts }
Node     { ID, Name, Op, JitKernel, Inputs, Outputs, WeightToken, StreamID, SyncBarriers }
PortAllocation { PortID, BaseOffset, StrideExprs, PortType }
PortType { DType, ShapeSchema, LayoutSchema, SemanticKind, Constraints }
StrideFormula { Symbol, Multiplier }
```

Missing fields and types are tracked in `../puter/GAPS.md §3.2`. Adding them is a precondition for the rest of the compiler.

---

## 5. Banned patterns

In addition to the shared contract in `../puter/AGENTS.md`:

- **Model-named Go files in `runtime/`, `compiler/`, `optimizer/`, `codegen/`, `scheduler/`.** No `executor_diffusion.go`, no `latents.go`, no `flux_layout.go`. The closest exceptions are generic `scheduler.go`-style files that implement ARCHITECTURE.md §4.4's DAG scheduler — those are not "noise schedulers".
- **Step-name cases in `executor.go` like `case "diffusion.prepare_latents":`** or `case "scheduler.timesteps":`. Atomic step kinds are domain-prefixed by op family.
- **Old-format YAMLs** (no leading `kind:`). Migrate the 13 known ones; do not add new ones.
- **Map-based lookups in execution-loop code.** Once the executor is rewritten per `../puter/GAPS.md` P1, all port/buffer/kernel handles are pre-resolved into flat slices (ARCHITECTURE.md §5.2). No `map[K]V` reads in the dispatch loop.
- **`make([]byte, ...)` for the workspace.** Workspace is off-Go-heap (ARCHITECTURE.md §5.2).

---

## 6. Mechanical enforcement

`scripts/check_banned.sh` enforces §1, §3, §5. Run via `make check`. Use `make verify` (check + test) before declaring work done.

If a rule needs to change, update this AGENTS.md *and* update `scripts/check_banned.sh` *and* note the change in the commit message. Do not edit the script to make your change pass.

---

## 7. Reading order

1. `../puter/ARCHITECTURE.md` — the spec.
2. `../puter/AGENTS.md` — the shared coding contract.
3. `../puter/GAPS.md` — what's done, what's not.
4. This document.
5. The package(s) directly relevant to the task.

If something in the existing code looks wrong, read it before concluding it is wrong. The user is building toward a goal; existing structure is usually load-bearing.

## 8. Definition of Done

Same as `../puter/AGENTS.md §2`. Paste the output.
