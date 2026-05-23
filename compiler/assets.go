package compiler

import (
	"context"
	"fmt"
	"io/fs"
	"strings"

	"gopkg.in/yaml.v3"

	hfconfig "github.com/theapemachine/hf/config"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
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
	flattened, err := compiler.flattenIncludes(ctx, input, assetFS)

	if err != nil {
		return nil, err
	}

	programYAML, err := yaml.Marshal(flattened)

	if err != nil {
		return nil, fmt.Errorf("manifest include: marshal flattened program: %w", err)
	}

	input.ProgramYAML = programYAML

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

	for includeName, includeObject := range output.Program.IncludeObjects {
		graph, computeGraph, compileErr := compiler.compileManifestObject(ctx, includeObject, input.CacheDir)

		if compileErr != nil {
			return nil, newError(includeName, "compile", fmt.Sprintf("include object %q", includeName), compileErr)
		}

		if graph != nil {
			output.Graphs[includeName] = graph
		}

		if computeGraph != nil {
			output.ComputeGraphs[includeName] = computeGraph
		}

		if err := compiler.compileNestedManifestObjects(ctx, assetFS, includeName, includeObject, input.CacheDir, output); err != nil {
			return nil, err
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

func (compiler *Compiler) compileManifestObject(
	ctx context.Context,
	includeObject any,
	cacheDir string,
) (*ast.Graph, *ir.Graph, error) {
	document, ok := includeObject.(map[string]any)

	if !ok {
		return nil, nil, nil
	}

	if err := compiler.enrichDocumentFromHub(ctx, document, cacheDir); err != nil {
		return nil, nil, err
	}

	raw, err := yaml.Marshal(document)

	if err != nil {
		return nil, nil, err
	}

	expandVars, err := diffusionRecipeConfig(document)

	if err != nil {
		expandVars = nil
	}

	return compiler.compileAnyManifestDocument(ctx, raw, cacheDir, expandVars)
}

func (compiler *Compiler) compileNestedManifestObjects(
	ctx context.Context,
	assetFS fs.FS,
	includeName string,
	includeObject any,
	cacheDir string,
	output *CompileOutput,
) error {
	document, ok := includeObject.(map[string]any)

	if !ok {
		return nil
	}

	runtime, ok := nestedMap(document, "system", "runtime")

	if !ok {
		return nil
	}

	if err := compiler.enrichDocumentFromHub(ctx, document, cacheDir); err != nil {
		return err
	}

	for componentName, value := range runtime {
		component, ok := value.(map[string]any)

		if !ok {
			continue
		}

		manifestPath, ok := component["manifest"].(string)

		if !ok || manifestPath == "" {
			continue
		}

		graphName := includeName + "." + componentName
		componentVariables := compiler.componentHubVariables(ctx, component, cacheDir)
		expandVars := mergeComponentExpandVariables(document, component, componentVariables)

		graph, computeGraph, err := compiler.compileModelIncludeWithVariables(
			ctx,
			assetFS,
			manifestPath,
			cacheDir,
			expandVars,
		)

		if err != nil {
			return newError(graphName, "compile", "nested manifest include", err)
		}

		output.Graphs[graphName] = graph

		if computeGraph != nil {
			output.ComputeGraphs[graphName] = computeGraph
		}
	}

	return nil
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

	return compiler.compileAnyManifestDocument(ctx, raw, cacheDir, nil)
}

func (compiler *Compiler) compileModelIncludeWithVariables(
	ctx context.Context,
	assetFS fs.FS,
	includePath string,
	cacheDir string,
	expandVars map[string]any,
) (*ast.Graph, *ir.Graph, error) {
	raw, err := fs.ReadFile(assetFS, NormalizeIncludePath(includePath))

	if err != nil {
		return nil, nil, fmt.Errorf("read model include %q: %w", includePath, err)
	}

	return compiler.compileModelDocument(ctx, raw, cacheDir, expandVars)
}

func (compiler *Compiler) compileAnyManifestDocument(
	ctx context.Context,
	raw []byte,
	cacheDir string,
	expandVars map[string]any,
) (*ast.Graph, *ir.Graph, error) {
	graph, computeGraph, err := compiler.compileTopologyDocument(raw)

	if err == nil {
		return graph, computeGraph, nil
	}

	return compiler.compileModelDocument(ctx, raw, cacheDir, expandVars)
}

func (compiler *Compiler) compileTopologyDocument(raw []byte) (*ast.Graph, *ir.Graph, error) {
	topology := &ast.Topology{}

	if err := yaml.Unmarshal(raw, topology); err != nil {
		return nil, nil, err
	}

	if len(topology.Nodes) == 0 {
		return nil, nil, fmt.Errorf("manifest topology: no nodes")
	}

	topology, err := compiler.expander.ExpandTopology(topology)

	if err != nil {
		return nil, nil, err
	}

	graph, err := compiler.topology.Topology(topology, dtype.Float32)

	if err != nil {
		return nil, nil, err
	}

	computeGraph, err := compiler.compute.Graph(graph)

	if err != nil {
		return nil, nil, err
	}

	return graph, computeGraph, nil
}

func nestedMap(document map[string]any, path ...string) (map[string]any, bool) {
	current := any(document)

	for _, segment := range path {
		values, ok := current.(map[string]any)

		if !ok {
			return nil, false
		}

		current, ok = values[segment]

		if !ok {
			return nil, false
		}
	}

	values, ok := current.(map[string]any)

	return values, ok
}

func (compiler *Compiler) compileModelDocument(
	ctx context.Context,
	raw []byte,
	cacheDir string,
	expandVars map[string]any,
) (*ast.Graph, *ir.Graph, error) {
	block, err := parse.BlockModelFromYAML(raw)

	if err != nil {
		return nil, nil, err
	}

	spec := block.FromSafeTensorsSpec()

	if spec != nil {
		return compiler.compileFromSafeTensors(ctx, spec, cacheDir, expandVars)
	}

	topology, err := block.TopologyAST()

	if err != nil {
		return nil, nil, err
	}

	topology, err = compiler.expander.ExpandTopologyWithVariables(topology, expandVars)

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
	expandVars map[string]any,
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

	if rawVariables, ok := spec["variables"].(map[string]any); ok {
		config = mergeConfigMaps(config, rawVariables)
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

	subfolder := safetensorsSubfolder(filename)

	if subfolder != "" {
		hubConfig, hubErr := compiler.resolver.ComponentConfig(ctx, location, subfolder, cacheDir)

		if hubErr == nil {
			config = mergeConfigMaps(hubConfig, config)
		}
	}

	if expandVars != nil {
		config = mergeConfigMaps(config, expandVars)
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
