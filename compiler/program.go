package compiler

import (
	"context"
	"fmt"
	"io/fs"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/typer"
)

/*
CompileInput is the host-provided context for one program compilation.
*/
type CompileInput struct {
	ProgramYAML []byte
	CacheDir    string
}

/*
CompileOutput is the compiled program and named compute graphs.

Workspaces is the planner's output for each graph that was successfully
typed and planned during CompileAssets. The map is keyed by the same
graph name CompileOutput.Graphs uses. Empty when the planner is not
enabled (the default during the staged Phase 1.2 rollout) or when a
given graph could not be planned (e.g., the typer left an edge with
dtype.Invalid).
*/
type CompileOutput struct {
	Program       *ast.Program
	Graphs        map[string]*ast.Graph
	ComputeGraphs map[string]*dag.Graph
	Workspaces    map[string]*ir.Topology
}

/*
Pool resolves manifest assets and Hugging Face repositories for program compilation.
*/
type Pool struct {
	catalog catalog.Catalog
}

/*
NewPool constructs a compiler asset pool.
*/
func NewPool(catalogInstance catalog.Catalog) *Pool {
	return &Pool{catalog: catalogInstance}
}

/*
ProgramCompiler parses and compiles manifest program YAML into runtime IR.
Graph lowering is filled in as the ARCHITECTURE.md pipeline lands in manifesto.
*/
type ProgramCompiler struct {
	pool             *Pool
	parser           *parse.Parser
	resolver         IncludeResolver
	typerOptions     typer.Options
	skipTyper        bool
	optimizerOptions optimizer.Options
	skipOptimizer    bool
	codegenOptions   codegen.EmitOptions
	skipCodegen      bool
	plannerBindings  ir.SymbolMap
}

/*
NewProgramCompiler constructs a program compiler from host-provided dependencies.
*/
func NewProgramCompiler(pool *Pool) (*ProgramCompiler, error) {
	if pool == nil {
		return nil, fmt.Errorf("compiler: asset pool is required")
	}

	return &ProgramCompiler{
		pool:   pool,
		parser: parse.NewParser(),
	}, nil
}

/*
WithIncludeResolver injects a resolver used to materialize block YAML for
`include:` references in a program manifest. Without a resolver, includes are
silently skipped — programs that reference only host ops (no graph.call) still
compile.
*/
func (programCompiler *ProgramCompiler) WithIncludeResolver(resolver IncludeResolver) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.resolver = resolver

	return programCompiler
}

/*
WithOptimizerOptions overrides the optimizer pipeline configuration used by
CompileAssets. Callers can disable individual passes or supply a custom
TileTarget for non-default CPU profiles.
*/
func (programCompiler *ProgramCompiler) WithOptimizerOptions(options optimizer.Options) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.optimizerOptions = options

	return programCompiler
}

/*
DisableOptimizer skips the optimizer pipeline entirely. Used by tests that
need to assert on un-rewritten graphs.
*/
func (programCompiler *ProgramCompiler) DisableOptimizer() *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.skipOptimizer = true

	return programCompiler
}

/*
WithTyperOptions overrides the typer pipeline configuration applied to
every lowered topology. Used by tests and by callers who need to inspect
edge errors before adaptor synthesis runs.
*/
func (programCompiler *ProgramCompiler) WithTyperOptions(options typer.Options) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.typerOptions = options

	return programCompiler
}

/*
DisableTyper skips Phase 2.2 unification + Phase 2.3 adaptor synthesis.
Used by tests that need to assert on the raw lowered graph.
*/
func (programCompiler *ProgramCompiler) DisableTyper() *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.skipTyper = true

	return programCompiler
}

/*
WithCodegenOptions overrides the kernel-emission options applied to fused
nodes. Callers can restrict targets (e.g. only TargetCPU on machines
without Metal).
*/
func (programCompiler *ProgramCompiler) WithCodegenOptions(options codegen.EmitOptions) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.codegenOptions = options

	return programCompiler
}

/*
DisableCodegen skips the kernel-emission pass. Useful for tests that
assert on un-attached FuseOp nodes or when only the optimizer's structural
output is needed.
*/
func (programCompiler *ProgramCompiler) DisableCodegen() *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.skipCodegen = true

	return programCompiler
}

/*
WithPlannerBindings injects runtime-supplied SymbolMap entries (max
sequence length, max batch size, etc.) into the static memory planner.
These merge with the typer's edge-unified bindings so PortByteSize can
size every dynamic dimension at compile time.

The planner itself always runs as part of CompileAssets — it produces
the workspace layout the runtime executor depends on
(ARCHITECTURE.md §5.1 / §6). Callers that need to size dynamic
dimensions must supply bindings here, or the planner will reject the
graph with an "unbound symbol" diagnostic.
*/
func (programCompiler *ProgramCompiler) WithPlannerBindings(bindings ir.SymbolMap) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.plannerBindings = bindings

	return programCompiler
}

/*
CompileAssets parses one program manifest, resolves its include directives,
lowers each included topology into both ast.Graph and dag.Graph
representations, and returns the assembled CompileOutput.
*/
func (programCompiler *ProgramCompiler) CompileAssets(
	ctx context.Context,
	input CompileInput,
	assetFS fs.FS,
) (*CompileOutput, error) {
	_ = assetFS
	_ = input.CacheDir

	if len(input.ProgramYAML) == 0 {
		return nil, fmt.Errorf("compiler: program yaml is required")
	}

	program, err := programCompiler.parser.Program(input.ProgramYAML)

	if err != nil {
		return nil, fmt.Errorf("compiler: parse program: %w", err)
	}

	graphs := make(map[string]*ast.Graph, len(program.Includes))
	computeGraphs := make(map[string]*dag.Graph, len(program.Includes))

	for name, source := range program.Includes {
		if programCompiler.resolver == nil {
			return nil, fmt.Errorf(
				"compiler: include %q references %q but no IncludeResolver is configured",
				name, source,
			)
		}

		blockYAML, err := programCompiler.resolver.ResolveInclude(ctx, IncludeSource{
			Name:   name,
			Source: source,
		})

		if err != nil {
			return nil, &ResolverError{Include: name, Cause: err}
		}

		lowered, err := lowerBlock(name, blockYAML)

		if err != nil {
			return nil, &ResolverError{Include: name, Cause: err}
		}

		if !programCompiler.skipTyper {
			if _, err := typer.Run(lowered.AST, programCompiler.typerOptions); err != nil {
				return nil, &ResolverError{Include: name, Cause: err}
			}
		}

		if !programCompiler.skipOptimizer {
			if _, err := optimizer.Run(lowered.AST, programCompiler.optimizerOptions); err != nil {
				return nil, &ResolverError{Include: name, Cause: err}
			}
		}

		if !programCompiler.skipCodegen {
			if _, err := codegen.AttachKernels(lowered.AST, programCompiler.codegenOptions); err != nil {
				return nil, &ResolverError{Include: name, Cause: err}
			}
		}

		graphs[name] = lowered.AST
		computeGraphs[name] = lowered.DAG
	}

	workspaces, err := programCompiler.runPlanner(graphs)

	if err != nil {
		return nil, err
	}

	return &CompileOutput{
		Program:       program,
		Graphs:        graphs,
		ComputeGraphs: computeGraphs,
		Workspaces:    workspaces,
	}, nil
}

/*
runPlanner is the post-typer planner pass. Every typed graph is
converted into an *ir.Topology via TopologyForPlanning and then planned
via ir.PlanWorkspace; the resulting topology — with WorkspaceLayout
populated — is returned keyed by graph name.

The planner is bypassed when the typer was disabled: typing is a hard
prerequisite (PortByteSize needs Port.Type.DType and ShapeSchema
populated, both of which the typer fills in), so calling the planner
on untyped graphs would just throw "dtype Invalid" errors. Tests that
DisableTyper are exercising structural lowering in isolation and don't
need workspace output. Production paths (the orchestrator) never
disable the typer, so this carve-out doesn't affect runtime behaviour.

Caller-supplied bindings are merged with the typer's edge-unified
bindings so runtime dimension bounds (max_seq_len, max_batch, etc.) and
compile-time bindings both feed the planner. Conflicts surface as
panics from mergeSymbolMaps because the typer has already caught any
contradiction long before the planner sees the graph.

Failures are wrapped with the graph name so the diagnostic points at
the offending include directly.
*/
func (programCompiler *ProgramCompiler) runPlanner(
	graphs map[string]*ast.Graph,
) (map[string]*ir.Topology, error) {
	if programCompiler.skipTyper {
		return nil, nil
	}

	workspaces := make(map[string]*ir.Topology, len(graphs))

	for name, graph := range graphs {
		mergedBindings := mergeSymbolMaps(graph.Bindings, programCompiler.plannerBindings)
		graph.Bindings = mergedBindings

		topology, err := PlanGraph(graph)

		if err != nil {
			return nil, fmt.Errorf("compiler: plan graph %q: %w", name, err)
		}

		workspaces[name] = topology
	}

	return workspaces, nil
}

/*
mergeSymbolMaps returns a new SymbolMap containing every binding from
both inputs. When the same symbol appears on both sides with the same
value it's idempotent; conflicting values surface as a panic because the
typer is supposed to have caught conflicts long before the planner runs
(see ir.bindSymbol). A panic here means a real invariant violation —
either the typer missed something or the caller passed bindings that
contradict the manifest.
*/
func mergeSymbolMaps(base, overlay ir.SymbolMap) ir.SymbolMap {
	merged := make(ir.SymbolMap, len(base)+len(overlay))

	for symbol, value := range base {
		merged[symbol] = value
	}

	for symbol, value := range overlay {
		if existing, ok := merged[symbol]; ok && existing != value {
			panic(fmt.Sprintf(
				"compiler: conflicting binding for symbol %q: %d (typer) vs %d (caller)",
				symbol, existing, value,
			))
		}

		merged[symbol] = value
	}

	return merged
}

/*
lowerBlock parses one block YAML payload into an ast.Topology and lowers it
into the LoweredGraph pair. Block payloads must contain a topology section
either at the document root or under `system.topology`.
*/
func lowerBlock(name string, blockYAML []byte) (*LoweredGraph, error) {
	block, err := parse.BlockModelFromYAML(blockYAML)

	if err != nil {
		return nil, fmt.Errorf("parse block %q: %w", name, err)
	}

	topology, err := block.TopologyAST()

	if err != nil {
		return nil, fmt.Errorf("topology for block %q: %w", name, err)
	}

	return LowerTopology(topology)
}
