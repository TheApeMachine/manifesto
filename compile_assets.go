package manifest

import (
	"context"
	"fmt"
	"io/fs"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/hfconfig"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/resolve"
)

/*
CompileAssets parses a program manifest and compiles every referenced model
include and program graph module into ast.Graph and ir.Graph maps.
*/
func (compiler *Compiler) CompileAssets(
	ctx context.Context,
	input CompileInput,
	assetFS fs.FS,
) (*CompileOutput, error) {
	output, err := compiler.Compile(ctx, input)

	if err != nil {
		return nil, err
	}

	if output.Program == nil {
		return output, nil
	}

	for includeName, includePath := range output.Program.Includes {
		graph, computeGraph, compileErr := compiler.compileProgramInclude(ctx, assetFS, includePath, input.CacheDir)

		if compileErr != nil {
			return nil, newError(includePath, "compile", fmt.Sprintf("include %q", includeName), compileErr)
		}

		output.Graphs[includeName] = graph

		if computeGraph != nil {
			output.ComputeGraphs[includeName] = computeGraph
		}
	}

	for graphName, module := range output.Program.Graphs {
		if module.Topology == nil {
			continue
		}

		graph, computeGraph, compileErr := compiler.compileTopologyModule(
			ctx,
			graphName,
			module,
			input,
		)

		if compileErr != nil {
			return nil, compileErr
		}

		output.Graphs[graphName] = graph

		if computeGraph != nil {
			output.ComputeGraphs[graphName] = computeGraph
		}
	}

	return output, nil
}

func (compiler *Compiler) compileProgramInclude(
	ctx context.Context,
	assetFS fs.FS,
	includePath string,
	cacheDir string,
) (*ast.Graph, *ir.Graph, error) {
	repoID, componentName, ok := strings.Cut(strings.TrimPrefix(includePath, "hf://"), "#")

	if strings.HasPrefix(includePath, "hf://") && ok {
		location := resolve.RepoLocation{
			RepoID:   repoID,
			RepoType: resolve.ModelRepo,
			Revision: "main",
		}

		component := ast.Component{
			ClassName: componentName,
			Subfolder: componentName,
		}

		componentGraph, computeGraph, err := compiler.compileComponent(
			ctx,
			location,
			cacheDir,
			component,
		)

		if err != nil {
			return nil, nil, err
		}

		return componentGraph.Graph, computeGraph, nil
	}

	return compiler.compileModelInclude(ctx, assetFS, includePath, cacheDir)
}

func (compiler *Compiler) compileModelInclude(
	ctx context.Context,
	assetFS fs.FS,
	includePath string,
	cacheDir string,
) (*ast.Graph, *ir.Graph, error) {
	var raw []byte
	var err error

	if strings.HasPrefix(includePath, "hf://") {
		repoID := strings.TrimPrefix(includePath, "hf://")
		location := resolve.RepoLocation{
			RepoID:   repoID,
			RepoType: resolve.ModelRepo,
			Revision: "main",
		}

		reader, _, err := compiler.resolver.Open(ctx, location, "config.json", cacheDir)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to open config.json for %s: %w", repoID, err)
		}
		defer reader.Close()

		config, err := hfconfig.ParseConfig(reader)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to parse config.json for %s: %w", repoID, err)
		}

		yamlStr, err := hfconfig.GenerateYAML(config, repoID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to generate YAML for %s: %w", repoID, err)
		}
		raw = []byte(yamlStr)
	} else {
		raw, err = fs.ReadFile(assetFS, includePath)
		if err != nil {
			return nil, nil, fmt.Errorf("read model include %q: %w", includePath, err)
		}
	}

	block, err := parse.BlockModelFromYAML(raw)

	if err != nil {
		return nil, nil, err
	}

	spec := block.FromSafeTensorsSpec()

	if spec != nil {
		return compiler.compileFromSafeTensors(ctx, spec, cacheDir)
	}

	topology, err := block.TopologyAST()

	if err != nil {
		return nil, nil, err
	}

	topology, err = compiler.expander.ExpandTopology(topology)

	if err != nil {
		return nil, nil, err
	}

	executionDType := dtype.Float32

	graph, err := compiler.topology.Topology(topology, executionDType)

	if err != nil {
		return nil, nil, err
	}

	repoID := block.PrimaryRepoID()

	if repoID != "" {
		location := resolve.RepoLocation{
			RepoID:   repoID,
			RepoType: resolve.ModelRepo,
			Revision: "main",
		}

		if block.System.Runtime.Model.Revision != "" {
			location.Revision = block.System.Runtime.Model.Revision
		}

		filenames, weightErr := compiler.resolver.WeightFiles(
			ctx,
			location,
			block.WeightSubfolder(),
			cacheDir,
		)

		if weightErr == nil {
			weightPath, bindErr := compiler.bindWeightsFiles(ctx, location, cacheDir, filenames, graph, nil)

			if bindErr != nil {
				return nil, nil, bindErr
			}

			if graph.Metadata == nil {
				graph.Metadata = make(map[string]any)
			}

			graph.Metadata["weights_path"] = weightPath
		}
	}

	computeGraph, err := compiler.compute.Graph(graph)

	if err != nil {
		return nil, nil, err
	}

	return graph, computeGraph, nil
}

func (compiler *Compiler) compileFromSafeTensors(
	ctx context.Context,
	spec map[string]any,
	cacheDir string,
) (*ast.Graph, *ir.Graph, error) {
	repoID, _ := spec["source"].(string)

	if repoID == "" {
		return nil, nil, fmt.Errorf("from_safetensors: source is required")
	}

	className, _ := spec["architecture"].(string)

	if className == "" {
		return nil, nil, fmt.Errorf("from_safetensors: architecture is required")
	}

	config, _ := spec["config"].(map[string]any)

	if config == nil {
		config = make(map[string]any)
	}

	recipe, err := compiler.registry.Recipe(className)

	if err != nil {
		return nil, nil, err
	}

	topology, err := compiler.expander.Topology(recipe, config)

	if err != nil {
		return nil, nil, err
	}

	executionDType, err := compiler.resolver.ExecutionDType(config)

	if err != nil {
		executionDType = dtype.Float32
	}

	graph, err := compiler.topology.Topology(topology, executionDType)

	if err != nil {
		return nil, nil, err
	}

	location := resolve.RepoLocation{
		RepoID:   repoID,
		RepoType: resolve.ModelRepo,
		Revision: "main",
	}

	filename, _ := spec["file"].(string)

	if filename == "" {
		filename = "model.safetensors"
	}

	weightPath, bindErr := compiler.bindWeightsFile(ctx, location, cacheDir, filename, graph, recipe.WeightMap)

	if bindErr != nil {
		return nil, nil, bindErr
	}

	if graph.Metadata == nil {
		graph.Metadata = make(map[string]any)
	}

	graph.Metadata["weights_path"] = weightPath

	computeGraph, err := compiler.compute.Graph(graph)

	if err != nil {
		return nil, nil, err
	}

	return graph, computeGraph, nil
}

func (compiler *Compiler) compileTopologyModule(
	ctx context.Context,
	graphName string,
	module ast.GraphModule,
	input CompileInput,
) (*ast.Graph, *ir.Graph, error) {
	executionDType := dtype.Float32

	topology, err := compiler.expander.ExpandTopology(module.Topology)
	if err != nil {
		return nil, nil, newError(graphName, "expand", "expand program graph module", err)
	}

	graph, err := compiler.topology.Topology(topology, executionDType)

	if err != nil {
		return nil, nil, newError(graphName, "lower", "lower program graph module", err)
	}

	if input.Repo.RepoID != "" {
		component := ast.Component{Subfolder: graphName}

		if bindErr := compiler.bindComponentWeights(ctx, input.Repo, input.CacheDir, component, graph, nil); bindErr != nil {
			return nil, nil, bindErr
		}
	}

	if graph.Metadata == nil {
		graph.Metadata = make(map[string]any)
	}

	computeGraph, err := compiler.compute.Graph(graph)

	if err != nil {
		return nil, nil, newError(graphName, "ir", "lower compute graph", err)
	}

	return graph, computeGraph, nil
}

func (compiler *Compiler) bindWeightsFiles(
	ctx context.Context,
	location resolve.RepoLocation,
	cacheDir string,
	filenames []string,
	graph *ast.Graph,
	weightMap map[string]string,
) (string, error) {
	firstWeightPath := ""

	for _, filename := range filenames {
		weightPath, err := compiler.bindWeightsFile(ctx, location, cacheDir, filename, graph, weightMap)

		if err != nil {
			return "", err
		}

		if firstWeightPath == "" {
			firstWeightPath = weightPath
		}
	}

	return firstWeightPath, nil
}

func (compiler *Compiler) bindWeightsFile(
	ctx context.Context,
	location resolve.RepoLocation,
	cacheDir string,
	filename string,
	graph *ast.Graph,
	weightMap map[string]string,
) (string, error) {
	reader, file, err := compiler.resolver.Open(ctx, location, filename, cacheDir)

	if err != nil {
		return "", err
	}

	defer reader.Close()

	index, err := compiler.binder.Index(reader)

	if err != nil {
		return "", err
	}

	if err := compiler.binder.Bind(graph, index, weightMap); err != nil {
		return "", err
	}

	compiler.markWeightFile(graph, weightNames(index), file.Path)

	return file.Path, nil
}

func (compiler *Compiler) markWeightFile(
	graph *ast.Graph,
	weightNames map[string]struct{},
	weightPath string,
) {
	for _, node := range graph.Nodes {
		if node.Weights == nil {
			continue
		}

		if _, ok := weightNames[node.Weights.TensorName]; !ok {
			continue
		}

		if node.Metadata == nil {
			node.Metadata = make(map[string]any)
		}

		node.Metadata["weight_file"] = weightPath
	}
}

func weightNames[T any](index map[string]T) map[string]struct{} {
	names := make(map[string]struct{}, len(index))

	for name := range index {
		names[name] = struct{}{}
	}

	return names
}

// NormalizeIncludePath maps template-relative paths for asset FS reads.
func NormalizeIncludePath(name string) string {
	trimmed := strings.TrimSpace(name)

	if trimmed == "" {
		return trimmed
	}

	if strings.HasPrefix(trimmed, "hf://") {
		return trimmed
	}

	if strings.Contains(trimmed, "/") {
		return trimmed
	}

	return parse.ResolveIncludePath(trimmed)
}
