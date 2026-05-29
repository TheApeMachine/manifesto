package compiler

import (
	"context"
	"fmt"
	"io/fs"

	"github.com/theapemachine/errnie"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/typer"
	"github.com/theapemachine/manifesto/types"
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
graph name CompileOutput.Graphs uses.
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
*/
type ProgramCompiler struct {
	ctx               context.Context
	cancel            context.CancelFunc
	pool              *Pool
	parser            *parse.Parser
	resolver          IncludeResolver
	operationRegistry *types.OperationRegistry
	typerOptions      typer.Options
	skipTyper         bool
	optimizerOptions  optimizer.Options
	skipOptimizer     bool
	codegenOptions    codegen.EmitOptions
	skipCodegen       bool
	plannerBindings   ir.SymbolMap
	streamSchedule    ir.StreamScheduleOptions
}

/*
NewProgramCompiler constructs a program compiler from host-provided dependencies.
*/
func NewProgramCompiler(ctx context.Context, pool *Pool) (*ProgramCompiler, error) {
	ctx, cancel := context.WithCancel(ctx)

	registry, err := types.NewOperationRegistry()

	if err != nil {
		cancel()
		return nil, fmt.Errorf("compiler: operation registry: %w", err)
	}

	programCompiler := &ProgramCompiler{
		ctx:               ctx,
		cancel:            cancel,
		pool:              pool,
		parser:            parse.NewParser(),
		operationRegistry: registry,
	}

	return programCompiler, errnie.Require(map[string]any{
		"ctx":               programCompiler.ctx,
		"cancel":            programCompiler.cancel,
		"pool":              pool,
		"operationRegistry": programCompiler.operationRegistry,
	})
}

/*
WithIncludeResolver injects a resolver used to materialize block YAML for
`include:` references in a program manifest. CompileAssets returns an error
when a program declares includes but no resolver is configured.
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
*/
func (programCompiler *ProgramCompiler) WithPlannerBindings(bindings ir.SymbolMap) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.plannerBindings = bindings

	return programCompiler
}

/*
WithStreamSchedule configures ARCHITECTURE.md §4.4 stream partitioning on
the planned topology. Zero MaxStreams lets the scheduler use the topology
width as the stream cap.
*/
func (programCompiler *ProgramCompiler) WithStreamSchedule(options ir.StreamScheduleOptions) *ProgramCompiler {
	if programCompiler == nil {
		return nil
	}

	programCompiler.streamSchedule = options

	return programCompiler
}

/*
CompileAssets parses one program manifest, resolves its include directives,
lowers each included topology, runs the typer → optimizer → codegen pipeline,
materializes a fresh dag.Graph from the final ast.Graph, and plans workspace
layout for every typed graph.
*/
func (programCompiler *ProgramCompiler) CompileAssets(
	ctx context.Context,
	input CompileInput,
	assetFS fs.FS,
) (*CompileOutput, error) {
	_ = ctx
	_ = assetFS

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

		graph, computeGraph, err := programCompiler.compileIncludeGraph(name, blockYAML)

		if err != nil {
			return nil, &ResolverError{Include: name, Cause: err}
		}

		graphs[name] = graph
		computeGraphs[name] = computeGraph
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

func (programCompiler *ProgramCompiler) compileIncludeGraph(
	name string,
	blockYAML []byte,
) (*ast.Graph, *dag.Graph, error) {
	lowered, err := lowerBlock(name, blockYAML)

	if err != nil {
		return nil, nil, err
	}

	graph := lowered.AST

	if !programCompiler.skipTyper {
		if _, err := typer.Run(graph, programCompiler.typerOptions); err != nil {
			return nil, nil, err
		}
	}

	if !programCompiler.skipOptimizer {
		if _, err := optimizer.Run(graph, programCompiler.optimizerOptions); err != nil {
			return nil, nil, err
		}

		if !programCompiler.skipTyper {
			if _, err := typer.Run(graph, programCompiler.typerOptions); err != nil {
				return nil, nil, err
			}
		}
	}

	if !programCompiler.skipCodegen {
		if _, err := codegen.AttachKernels(graph, programCompiler.codegenOptions); err != nil {
			return nil, nil, err
		}
	}

	if err := validateGraphOps(graph, programCompiler.operationRegistry); err != nil {
		return nil, nil, err
	}

	computeGraph, err := BuildDAGFromGraph(graph)

	if err != nil {
		return nil, nil, err
	}

	return graph, computeGraph, nil
}

func (programCompiler *ProgramCompiler) runPlanner(
	graphs map[string]*ast.Graph,
) (map[string]*ir.Topology, error) {
	if programCompiler.skipTyper {
		return nil, nil
	}

	workspaces := make(map[string]*ir.Topology, len(graphs))

	for name, graph := range graphs {
		mergedBindings, err := mergeSymbolMaps(graph.Bindings, programCompiler.plannerBindings)

		if err != nil {
			return nil, fmt.Errorf("compiler: plan graph %q: %w", name, err)
		}

		graph.Bindings = mergedBindings

		topology, err := PlanGraph(graph, PlanGraphOptions{
			Registry:       programCompiler.operationRegistry,
			Bindings:       mergedBindings,
			Align:          64,
			StreamSchedule: programCompiler.streamSchedule,
		})

		if err != nil {
			return nil, fmt.Errorf("compiler: plan graph %q: %w", name, err)
		}

		workspaces[name] = topology
	}

	return workspaces, nil
}

func mergeSymbolMaps(base, overlay ir.SymbolMap) (ir.SymbolMap, error) {
	merged := make(ir.SymbolMap, len(base)+len(overlay))

	for symbol, value := range base {
		merged[symbol] = value
	}

	for symbol, value := range overlay {
		if existing, ok := merged[symbol]; ok && existing != value {
			return nil, fmt.Errorf(
				"compiler: conflicting binding for symbol %q: %d (typer) vs %d (caller)",
				symbol, existing, value,
			)
		}

		merged[symbol] = value
	}

	return merged, nil
}

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
