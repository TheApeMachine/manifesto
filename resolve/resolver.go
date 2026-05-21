package resolve

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

const modelIndexFile = "model_index.json"

/*
Resolver discovers Hugging Face pipeline layout and component configs through
a Hub implementation.
*/
type Resolver struct {
	hub Hub
}

/*
NewResolver constructs a Resolver backed by the supplied Hub.
*/
func NewResolver(hubClient Hub) *Resolver {
	return &Resolver{hub: hubClient}
}

/*
Pipeline reads model_index.json when present and returns component class names
and subfolder layout.
*/
func (resolver *Resolver) Pipeline(
	ctx context.Context,
	location RepoLocation,
	cacheDir string,
) (*ast.Pipeline, error) {
	var raw map[string]any

	err := resolver.hub.ReadJSON(ctx, location, modelIndexFile, cacheDir, &raw)

	if err != nil {
		return nil, fmt.Errorf("resolve pipeline: %w", err)
	}

	pipeline := &ast.Pipeline{
		ClassName:  resolver.stringValue(raw["_class_name"]),
		Components: make(map[string]ast.Component),
	}

	for name, value := range raw {
		if strings.HasPrefix(name, "_") {
			continue
		}

		component, ok := resolver.parseComponent(name, value)

		if !ok {
			continue
		}

		pipeline.Components[name] = component
	}

	return pipeline, nil
}

/*
ComponentConfig loads config.json or scheduler_config.json for one pipeline
subfolder.
*/
func (resolver *Resolver) ComponentConfig(
	ctx context.Context,
	location RepoLocation,
	subfolder string,
	cacheDir string,
) (map[string]any, error) {
	configNames := []string{
		fmt.Sprintf("%s/config.json", subfolder),
		fmt.Sprintf("%s/scheduler_config.json", subfolder),
	}

	var lastErr error

	for _, filename := range configNames {
		config := make(map[string]any)

		err := resolver.hub.ReadJSON(ctx, location, filename, cacheDir, &config)

		if err == nil {
			return config, nil
		}

		lastErr = err
	}

	return nil, fmt.Errorf("resolve component config: no config for %q: %w", subfolder, lastErr)
}

func (resolver *Resolver) parseComponent(name string, value any) (ast.Component, bool) {
	pair, ok := value.([]any)

	if !ok || len(pair) != 2 {
		return ast.Component{}, false
	}

	library, libraryOK := pair[0].(string)
	className, classOK := pair[1].(string)

	if !libraryOK || !classOK {
		return ast.Component{}, false
	}

	return ast.Component{
		Library:   library,
		ClassName: className,
		Subfolder: name,
	}, true
}

func (resolver *Resolver) stringValue(value any) string {
	text, ok := value.(string)

	if !ok {
		return ""
	}

	return text
}

/*
PrimaryWeightFile returns the first safetensors filename found for a component.
*/
func (resolver *Resolver) PrimaryWeightFile(
	ctx context.Context,
	location RepoLocation,
	subfolder string,
	cacheDir string,
) (string, error) {
	files, err := resolver.WeightFiles(ctx, location, subfolder, cacheDir)

	if err != nil {
		return "", err
	}

	return files[0], nil
}

/*
WeightFiles returns every safetensors filename found for a component.
*/
func (resolver *Resolver) WeightFiles(
	ctx context.Context,
	location RepoLocation,
	subfolder string,
	cacheDir string,
) ([]string, error) {
	candidates := []string{
		subfolder + "/model.safetensors",
		subfolder + "/diffusion_pytorch_model.safetensors",
	}

	for _, filename := range candidates {
		reader, _, err := resolver.hub.Open(ctx, location, filename, cacheDir)

		if err != nil {
			continue
		}

		reader.Close()

		return []string{filename}, nil
	}

	matches, err := resolver.hub.Glob(ctx, location, subfolder+"/*.safetensors", cacheDir)

	if err != nil {
		return nil, err
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("resolve weights: no safetensors in %q", subfolder)
	}

	return matches, nil
}

func (resolver *Resolver) ClassName(config map[string]any, fallback string) string {
	architectures, ok := config["architectures"].([]any)

	if ok && len(architectures) > 0 {
		className, classOK := architectures[0].(string)

		if classOK {
			return className
		}
	}

	className, ok := config["_class_name"].(string)

	if ok && className != "" {
		return className
	}

	return fallback
}

/*
ExecutionDType reads the component activation dtype from a Hugging Face config.
*/
func (resolver *Resolver) ExecutionDType(config map[string]any) (dtype.DType, error) {
	for _, key := range []string{"dtype", "torch_dtype"} {
		raw, ok := config[key]

		if !ok {
			continue
		}

		text, ok := raw.(string)

		if !ok {
			return dtype.Invalid, fmt.Errorf("resolve execution dtype: config %q is not a string", key)
		}

		if strings.EqualFold(strings.TrimSpace(text), "auto") {
			return dtype.Invalid, fmt.Errorf("resolve execution dtype: %q is auto; resolve weights first", key)
		}

		parsed, err := dtype.Parse(text)

		if err != nil {
			return dtype.Invalid, fmt.Errorf("resolve execution dtype from %q: %w", key, err)
		}

		return parsed, nil
	}

	return dtype.Float32, nil
}

/*
Open downloads and opens one repository file through the configured Hub.
*/
func (resolver *Resolver) Open(
	ctx context.Context,
	location RepoLocation,
	filename string,
	cacheDir string,
) (io.ReadCloser, *File, error) {
	return resolver.hub.Open(ctx, location, filename, cacheDir)
}
