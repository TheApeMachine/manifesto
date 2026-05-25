package runtime

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/resolve"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/typer"
	"github.com/theapemachine/manifesto/types"
)

/*
Orchestrator compiles and runs one manifest program end-to-end.
*/
type Orchestrator struct {
	hub             resolve.Hub
	parser          func(archive []byte) (types.Parser, error)
	compute         Backend
	host            HostOps
	stateMemory     tensor.Backend
	cacheDir        string
	stdin           io.Reader
	initialValues   map[string]any
	includeResolver compiler.IncludeResolver
	typerOptions    typer.Options
	typerConfigured bool
	plannerBindings ir.SymbolMap
}

/*
OrchestratorOptions configures an Orchestrator.
*/
type OrchestratorOptions struct {
	Hub             resolve.Hub
	Parser          func(archive []byte) (types.Parser, error)
	Compute         Backend
	Host            HostOps
	StateMemory     tensor.Backend
	CacheDir        string
	Stdin           io.Reader
	InitialValues   map[string]any
	IncludeResolver compiler.IncludeResolver

	// TyperOptions overrides the typer pipeline configuration applied to
	// every lowered topology. Leave the zero value to use the typer's
	// defaults (Infer + adaptor synthesis). Callers can set
	// DisableSynthesis when the runtime is wired against an execution
	// backend that does not yet replay synthesized adaptor nodes — see
	// the note in compiler.WithTyperOptions.
	TyperOptions typer.Options

	// ConfigureTyper, when true, applies TyperOptions to the compiler.
	// When false the field is ignored entirely, preserving the existing
	// "use the compiler defaults" behaviour.
	ConfigureTyper bool

	// PlannerBindings supplies runtime-decided dimension bounds (max
	// sequence length, max batch size, …) to the static memory planner
	// so PortByteSize can size every symbolic dimension at compile time.
	// Merged with the typer's compile-time bindings; conflicts panic
	// because the typer is supposed to have caught any contradiction
	// before the planner sees the graph (see compiler.mergeSymbolMaps).
	PlannerBindings ir.SymbolMap
}

/*
NewOrchestrator constructs an Orchestrator from host-provided dependencies.
*/
func NewOrchestrator(options OrchestratorOptions) (*Orchestrator, error) {
	if options.Hub == nil {
		return nil, fmt.Errorf("runtime orchestrator: hub is required")
	}

	if options.Parser == nil {
		return nil, fmt.Errorf("runtime orchestrator: parser is required")
	}

	if options.Compute == nil {
		return nil, fmt.Errorf("runtime orchestrator: compute backend is required")
	}

	if options.Host == nil {
		return nil, fmt.Errorf("runtime orchestrator: host ops are required")
	}

	stdin := options.Stdin

	if stdin == nil {
		stdin = os.Stdin
	}

	return &Orchestrator{
		hub:             options.Hub,
		parser:          options.Parser,
		compute:         options.Compute,
		host:            options.Host,
		stateMemory:     options.StateMemory,
		cacheDir:        options.CacheDir,
		stdin:           stdin,
		initialValues:   options.InitialValues,
		includeResolver: options.IncludeResolver,
		typerOptions:    options.TyperOptions,
		typerConfigured: options.ConfigureTyper,
		plannerBindings: options.PlannerBindings,
	}, nil
}

/*
Run loads, compiles, and executes one program manifest path.
*/
func (orchestrator *Orchestrator) Run(ctx context.Context, programPath string) error {
	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return fmt.Errorf("runtime orchestrator: read program %q: %w", programPath, err)
	}

	manifestCompiler, err := compiler.NewProgramCompiler(
		compiler.NewPool(catalog.NewFS(asset.TemplateFS())),
	)

	if err != nil {
		return fmt.Errorf("runtime orchestrator: new compiler: %w", err)
	}

	if orchestrator.includeResolver != nil {
		manifestCompiler = manifestCompiler.WithIncludeResolver(orchestrator.includeResolver)
	}

	if orchestrator.typerConfigured {
		manifestCompiler = manifestCompiler.WithTyperOptions(orchestrator.typerOptions)
	}

	if len(orchestrator.plannerBindings) > 0 {
		manifestCompiler = manifestCompiler.WithPlannerBindings(orchestrator.plannerBindings)
	}

	output, err := manifestCompiler.CompileAssets(ctx, compiler.CompileInput{
		ProgramYAML: programYAML,
		CacheDir:    orchestrator.cacheDir,
	}, asset.TemplateFS())

	if err != nil {
		return fmt.Errorf("runtime orchestrator: compile assets: %w", err)
	}

	if err := orchestrator.attachWorkspaces(output); err != nil {
		return fmt.Errorf("runtime orchestrator: attach workspaces: %w", err)
	}

	programSession, err := NewProgramSession(ProgramSessionOptions{
		Program:      output.Program,
		Graphs:       output.Graphs,
		Compute:      output.ComputeGraphs,
		Backend:      orchestrator.compute,
		Host:         orchestrator.host,
		Stdin:        orchestrator.stdin,
		StateBackend: orchestrator.stateMemory,
	})

	if err != nil {
		return fmt.Errorf("runtime orchestrator: new program session: %w", err)
	}

	if err := programSession.RunWithValues(ctx, orchestrator.initialValues); err != nil {
		return fmt.Errorf("runtime orchestrator: run program: %w", err)
	}

	return nil
}

/*
attachWorkspaces hands the planner output to the compute backend when
that backend advertises the WorkspaceAttacher capability. The backend
takes the typed *ast.Graph and the *ir.Topology with its populated
WorkspaceLayout, allocates the off-heap region, and pre-resolves each
node's input/output tensors so the dispatcher can index them directly
at call time instead of re-allocating per node.

Backends that do not implement WorkspaceAttacher (test mocks, future
remote/XLA backends with their own residency) skip this step silently;
the planner output stays available on CompileOutput.Workspaces for
those that consume it elsewhere.
*/
func (orchestrator *Orchestrator) attachWorkspaces(output *compiler.CompileOutput) error {
	if output == nil || len(output.Workspaces) == 0 {
		return nil
	}

	attacher, ok := orchestrator.compute.(WorkspaceAttacher)

	if !ok {
		return nil
	}

	for name, topology := range output.Workspaces {
		graph, hasGraph := output.Graphs[name]

		if !hasGraph {
			return fmt.Errorf("planner produced workspace for unknown graph %q", name)
		}

		if err := attacher.AttachWorkspace(name, graph, topology); err != nil {
			return fmt.Errorf("attach workspace for graph %q: %w", name, err)
		}
	}

	return nil
}
