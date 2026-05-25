package compiler

import (
	"context"
	"fmt"
	"io/fs"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/codegen"
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
*/
type CompileOutput struct {
	Program       *ast.Program
	Graphs        map[string]*ast.Graph
	ComputeGraphs map[string]*dag.Graph
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

	return &CompileOutput{
		Program:       program,
		Graphs:        graphs,
		ComputeGraphs: computeGraphs,
	}, nil
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
