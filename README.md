# manifesto

`github.com/theapemachine/manifesto` compiles YAML runtime programs and topology recipes into typed `ast.Graph` IR, scheduling DAGs, and planned workspace topologies.

## Compilation pipeline

Program manifests and included topology blocks converge on one graph pipeline:

```mermaid
flowchart LR
  YAML[program YAML] --> Parse[parse]
  Parse --> Program[ast.Program]
  Program --> Include[include resolve]
  Include --> Lower[lower topology]
  Lower --> AST[ast.Graph]
  AST --> Weights[weights.Binder]
  Weights --> Typer[typer]
  Typer --> Optimizer[optimizer]
  Optimizer --> Typer2[typer re-run]
  Typer2 --> Codegen[codegen]
  Codegen --> Validate[closed-world validate]
  Validate --> DAG[dag.Graph]
  DAG --> Plan[ir.Topology planner]
```

1. **parse** — Load program YAML (`ast.Program`), resolve `$variables.*`, expand includes.
2. **lower** — Expand `repeat` templates and lower `ast.Topology` → `ast.Graph`.
3. **weights** — Bind SafeTensors checkpoint names onto graph nodes (optional, when a `types.Parser` is supplied).
4. **typer** — Hindley-Milner unification + adaptor synthesis (`shape.cast`, `shape.reshape`, `shape.transpose`).
5. **optimizer** — Fusion and other graph rewrites.
6. **codegen** — Attach CPU kernel metadata to fused nodes.
7. **validate** — Closed-world check against `types.OperationRegistry`.
8. **plan** — Static memory layout, I/O ports, stream scheduling → `ir.Topology`.

## Quick start

### Runtime program compilation

```go
import (
    "context"

    "github.com/theapemachine/manifesto/compiler"
    "github.com/theapemachine/manifesto/catalog"
)

pool := compiler.NewPool(catalog.NewFS(asset.TemplateFS()))
programCompiler, err := compiler.NewProgramCompiler(ctx, pool)
if err != nil { /* ... */ }

programCompiler = programCompiler.WithIncludeResolver(yourResolver)

out, err := programCompiler.CompileAssets(ctx, compiler.CompileInput{
    ProgramYAML: programBytes,
}, assetFS)
// out.Program, out.Graphs, out.ComputeGraphs, out.Workspaces
```

`WithWeightParser` binds checkpoint tensors before typing. `WithPlannerBindings` supplies runtime symbol values (`N`, batch caps) for workspace sizing.

### Checkpoint topology compilation

```go
checkpointCompiler, err := compiler.NewCompiler(ctx, pool, safetensorsParser)
if err != nil { /* ... */ }

checkpointCompiler = checkpointCompiler.WithTopology(topologyRecipe)

compiled, err := checkpointCompiler.CompileTopology()
// compiled.Graph, compiled.ComputeGraph

project, err := checkpointCompiler.Build()
// ir.Project with per-node weights for legacy planners
```

`CompileTopology` runs the same `CompileGraph` pipeline as `ProgramCompiler`. `Build` remains the weight-indexing path into `ir.Project`.

## Sub-packages

### [`compiler`](./compiler/)

Orchestrates the pipeline above. Primary types: `ProgramCompiler`, `CompileInput`, `CompileOutput`, `CompileGraph`, `CompiledGraph`. Legacy checkpoint binding: `Compiler`, `Build`, `CompileTopology`.

### [`ast`](./ast/)

Typed IR for the compiler and runtime:

- **Recipe / topology** — Block graphs, extends, weight maps, variable bindings
- **Graph** — Manifest-native compute IR (`Graph`, `GraphNode`, bound weights)
- **Program** — Runtime manifest: steps, loops, graph modules, schedulers, state

### [`catalog`](./catalog/)

Loads recipes, reusable blocks, and the architecture registry from an `io/fs.FS`. Hosts embed templates or mount an on-disk tree; `registry.yml` maps Hugging Face class names to recipe files.

### [`parse`](./parse/)

YAML loading for program manifests: includes, variables, `main` / `system.runtime` steps, and graph module definitions. Produces `ast.Program`.

### [`expand`](./expand/)

Materializes a recipe against a Hugging Face `config.json`:

- Merges `extends` chains from the catalog
- Variable interpolation and binding
- Repeat unrolling into a flat `ast.Topology`

### [`registry`](./registry/)

Looks up `ast.RegistryEntry` and loads the matching `ast.Recipe` for a resolved architecture class name.

### [`resolve`](./resolve/)

Hub-facing discovery: pipeline layout from `model_index.json`, component configs, execution dtype, primary SafeTensors file, and file open/read helpers. Defines `Hub`, `RepoLocation`, and download types.

### [`lower`](./lower/)

Shape inference and lowering from `ast.Topology` to `ast.Graph`, using the execution dtype from config.

### [`ir`](./ir/)

Lowers `ast.Graph` to compute IR (`ir.Graph`, `ir.Node`) with `tensor.Shape` on value types. Also provides operation IDs, required-op validation, and graph codec helpers.

### [`weights`](./weights/)

SafeTensors header indexing and binding checkpoint tensors to graph nodes according to the recipe weight map.

### [`runtime`](./runtime/)

Executes `ast.Program` steps against a `Backend` interface (`graph.call` and related ops). Device backends implement `Backend` outside this module.

### [`hfmodular`](./hfmodular/)

Transpiler stub for Hugging Face `modular_*.py` → recipe YAML. API is in place; transpilation is not implemented yet.

### [`dtype`](./dtype/)

Canonical numeric formats for the platform: `DType` enum, scalars, `Float16`, `BFloat16`, FP8, packed `Int4` and `Bool`. Wire format is little-endian everywhere.

### [`dtype/convert`](./dtype/convert/)

Scalar correctness paths for converting between dtypes (used before device upload when a backend does not accept a source dtype natively).

### [`tensor`](./tensor/)

Backend-neutral tensor abstraction: `Tensor` and `Backend` interfaces, host backend, tiered allocation (slab / mmap), NUMA hooks, arenas, sparse CSR, autograd tape types. See [`tensor/README.md`](./tensor/README.md) for allocator and lifecycle detail.

## Runtime execution

After compile, `runtime.Executor` walks program steps and delegates graph execution to your `runtime.Backend`. The compiler output’s `ComputeGraphs` map is what a production backend would schedule; wiring that backend is host-specific.

## Tests

```bash
go test ./...
```

Sub-packages use [GoConvey](https://github.com/smartystreets/goconvey) in several test files.

## Requirements

Go 1.26+ (`go.mod`).
