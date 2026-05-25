package runtime

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/resolve"
	"github.com/theapemachine/manifesto/tensor"
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

	output, err := manifestCompiler.CompileAssets(ctx, compiler.CompileInput{
		ProgramYAML: programYAML,
		CacheDir:    orchestrator.cacheDir,
	}, asset.TemplateFS())

	if err != nil {
		return fmt.Errorf("runtime orchestrator: compile assets: %w", err)
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
