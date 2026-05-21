package compiler

import (
	"context"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/expand"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/lower"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/registry"
	"github.com/theapemachine/manifesto/resolve"
	"github.com/theapemachine/manifesto/weights"
)

/*
CompileInput is the host-provided context for one compilation.
*/
type CompileInput struct {
	ProgramYAML []byte
	Repo        resolve.RepoLocation
	CacheDir    string
}

/*
CompileOutput is the compiled program, model bundle, and named graphs.
*/
type CompileOutput struct {
	Program       *ast.Program
	Model         *ast.ModelBundle
	Graphs        map[string]*ast.Graph
	ComputeGraphs map[string]*ir.Graph
	Registry      *registry.Registry
}

/*
Compiler resolves Hugging Face repositories and YAML programs into manifest and
compute IR. It composes catalog, parse, expand, lower, ir, resolve, and weight
binding stages.
*/
type Compiler struct {
	catalog  catalog.Catalog
	parser   *parse.Parser
	expander *expand.Recipe
	topology *lower.Lowerer
	compute  *ir.Lowerer
	binder   *weights.Binder
	resolver *resolve.Resolver
	registry *registry.Registry
}

/*
Options configures a Compiler.
*/
type Options struct {
	Catalog catalog.Catalog
	Hub     resolve.Hub
}

/*
NewCompiler constructs a Compiler from host-provided dependencies.
*/
func NewCompiler(options Options) (*Compiler, error) {
	registryInstance, err := registry.NewRegistry(options.Catalog)

	if err != nil {
		return nil, newError("", "registry", "initialize architecture registry", err)
	}

	return &Compiler{
		catalog:  options.Catalog,
		parser:   parse.NewParser(),
		expander: expand.NewRecipe(options.Catalog),
		topology: lower.NewLowerer(),
		compute:  ir.NewLowerer(),
		binder:   weights.NewBinder(),
		resolver: resolve.NewResolver(options.Hub),
		registry: registryInstance,
	}, nil
}

/*
Compile parses and resolves a program manifest against an optional repo.
*/
func (compiler *Compiler) Compile(ctx context.Context, input CompileInput) (*CompileOutput, error) {
	program, err := compiler.parser.Program(input.ProgramYAML)

	if err != nil {
		return nil, newError("", "parse", "parse program manifest", err)
	}

	output := &CompileOutput{
		Program:       program,
		Graphs:        make(map[string]*ast.Graph),
		ComputeGraphs: make(map[string]*ir.Graph),
		Registry:      compiler.registry,
	}

	if input.Repo.RepoID == "" {
		return output, nil
	}

	modelBundle, err := compiler.compileRepo(ctx, input.Repo, input.CacheDir, output)

	if err != nil {
		return nil, err
	}

	output.Model = modelBundle

	return output, nil
}

func (compiler *Compiler) compileRepo(
	ctx context.Context,
	location resolve.RepoLocation,
	cacheDir string,
	output *CompileOutput,
) (*ast.ModelBundle, error) {
	pipeline, err := compiler.resolver.Pipeline(ctx, location, cacheDir)

	if err != nil {
		return nil, newError(location.RepoID, "resolve", "discover pipeline", err)
	}

	bundle := &ast.ModelBundle{
		RepoID:     location.RepoID,
		Revision:   location.Revision,
		Pipeline:   pipeline,
		Components: make(map[string]*ast.ComponentGraph),
	}

	for name, component := range pipeline.Components {
		componentGraph, computeGraph, componentErr := compiler.compileComponent(
			ctx,
			location,
			cacheDir,
			component,
		)

		if componentErr != nil {
			return nil, componentErr
		}

		bundle.Components[name] = componentGraph

		if computeGraph != nil {
			output.ComputeGraphs[name] = computeGraph
		}
	}

	return bundle, nil
}

func (compiler *Compiler) compileComponent(
	ctx context.Context,
	location resolve.RepoLocation,
	cacheDir string,
	component ast.Component,
) (*ast.ComponentGraph, *ir.Graph, error) {
	config, err := compiler.resolver.ComponentConfig(ctx, location, component.Subfolder, cacheDir)

	if err != nil {
		return nil, nil, newError(location.RepoID, "resolve", "load component config", err)
	}

	component.Config = config
	className := compiler.resolver.ClassName(config, component.ClassName)

	executionDType, err := compiler.resolver.ExecutionDType(config)

	if err != nil {
		return nil, nil, newError(location.RepoID, "resolve", "resolve execution dtype", err)
	}

	recipe, err := compiler.registry.Recipe(className)

	if err != nil {
		return nil, nil, newError(location.RepoID, "registry", "resolve recipe", err)
	}

	topology, err := compiler.expander.Topology(recipe, config)

	if err != nil {
		return nil, nil, newError(location.RepoID, "expand", "expand recipe topology", err)
	}

	graph, err := compiler.topology.Topology(topology, executionDType)

	if err != nil {
		return nil, nil, newError(location.RepoID, "lower", "lower topology", err)
	}

	if bindErr := compiler.bindComponentWeights(ctx, location, cacheDir, component, graph, recipe.WeightMap); bindErr != nil {
		return nil, nil, bindErr
	}

	computeGraph, err := compiler.compute.Graph(graph)

	if err != nil {
		return nil, nil, newError(location.RepoID, "ir", "lower compute graph", err)
	}

	return &ast.ComponentGraph{
		ClassName:      className,
		ExecutionDType: executionDType,
		Graph:          graph,
	}, computeGraph, nil
}

func (compiler *Compiler) bindComponentWeights(
	ctx context.Context,
	location resolve.RepoLocation,
	cacheDir string,
	component ast.Component,
	graph *ast.Graph,
	weightMap map[string]string,
) error {
	filenames, err := compiler.resolver.WeightFiles(ctx, location, component.Subfolder, cacheDir)

	if err != nil {
		return newError(location.RepoID, "weights", "locate safetensors", err)
	}

	weightPath, err := compiler.bindWeightsFiles(ctx, location, cacheDir, filenames, graph, weightMap)

	if err != nil {
		return err
	}

	if graph.Metadata == nil {
		graph.Metadata = make(map[string]any)
	}

	graph.Metadata["weights_path"] = weightPath

	return nil
}
