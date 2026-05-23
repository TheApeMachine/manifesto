package compiler

import (
	"context"
	"fmt"
	"io/fs"
	"path"
	"strings"

	"gopkg.in/yaml.v3"

	hfconfig "github.com/theapemachine/hf/config"
	"github.com/theapemachine/manifesto/resolve"
)

/*
flattenIncludes expands include path references into inline YAML documents and
resolves dotted cross-references such as example.system.runtime.backend.
*/
func (compiler *Compiler) flattenIncludes(
	ctx context.Context,
	input CompileInput,
	assetFS fs.FS,
) (map[string]any, error) {
	root := make(map[string]any)

	if err := yaml.Unmarshal(input.ProgramYAML, &root); err != nil {
		return nil, fmt.Errorf("manifest include: parse root: %w", err)
	}

	if err := compiler.flattenManifest(ctx, root, assetFS, input.CacheDir, make(map[string]bool)); err != nil {
		return nil, err
	}

	resolveManifestReferences(root)

	return root, nil
}

func (compiler *Compiler) flattenManifest(
	ctx context.Context,
	document map[string]any,
	assetFS fs.FS,
	cacheDir string,
	seen map[string]bool,
) error {
	includes := includeMap(document)

	for name, value := range includes {
		location, ok := value.(string)

		if !ok {
			child, ok := value.(map[string]any)

			if ok {
				if err := compiler.flattenManifest(ctx, child, assetFS, cacheDir, seen); err != nil {
					return err
				}
			}

			continue
		}

		child, err := compiler.readIncludedManifest(ctx, location, assetFS, cacheDir, seen)

		if err != nil {
			return fmt.Errorf("manifest include %q: %w", name, err)
		}

	if err := compiler.enrichDocumentFromHub(ctx, child, cacheDir); err != nil {
			return fmt.Errorf("manifest include %q: %w", name, err)
		}

		includes[name] = child
	}

	return nil
}

func includeMap(document map[string]any) map[string]any {
	if includes, ok := document["include"].(map[string]any); ok {
		return includes
	}

	if includes, ok := document["includes"].(map[string]any); ok {
		return includes
	}

	return nil
}

func (compiler *Compiler) readIncludedManifest(
	ctx context.Context,
	location string,
	assetFS fs.FS,
	cacheDir string,
	seen map[string]bool,
) (map[string]any, error) {
	if strings.HasPrefix(location, "hf://") {
		return compiler.readHuggingFaceManifest(ctx, location, cacheDir)
	}

	filename := NormalizeIncludePath(location)

	if seen[filename] {
		return nil, fmt.Errorf("cycle at %q", location)
	}

	seen[filename] = true
	defer delete(seen, filename)

	raw, err := fs.ReadFile(assetFS, filename)

	if err != nil {
		return nil, fmt.Errorf("read %q: %w", filename, err)
	}

	document := make(map[string]any)

	if err := yaml.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("parse %q: %w", filename, err)
	}

	if err := compiler.flattenManifest(ctx, document, assetFS, cacheDir, seen); err != nil {
		return nil, err
	}

	return document, nil
}

func (compiler *Compiler) readHuggingFaceManifest(
	ctx context.Context,
	location string,
	cacheDir string,
) (map[string]any, error) {
	repoID, componentName, hasComponent := strings.Cut(strings.TrimPrefix(location, "hf://"), "#")
	repoLocation := resolve.RepoLocation{
		RepoID:   repoID,
		RepoType: resolve.ModelRepo,
		Revision: "main",
	}

	if hasComponent {
		return compiler.readHuggingFaceComponentManifest(ctx, repoLocation, componentName, cacheDir)
	}

	reader, _, err := compiler.resolver.Open(ctx, repoLocation, "config.json", cacheDir)

	if err != nil {
		return nil, fmt.Errorf("open config.json for %s: %w", repoID, err)
	}

	defer reader.Close()

	config, err := hfconfig.ParseConfig(reader)

	if err != nil {
		return nil, fmt.Errorf("parse config.json for %s: %w", repoID, err)
	}

	yamlString, err := hfconfig.GenerateYAML(config, repoID)

	if err != nil {
		return nil, fmt.Errorf("generate manifest for %s: %w", repoID, err)
	}

	document := make(map[string]any)

	if err := yaml.Unmarshal([]byte(yamlString), &document); err != nil {
		return nil, fmt.Errorf("parse generated manifest for %s: %w", repoID, err)
	}

	return document, nil
}

func (compiler *Compiler) readHuggingFaceComponentManifest(
	ctx context.Context,
	location resolve.RepoLocation,
	componentName string,
	cacheDir string,
) (map[string]any, error) {
	config, err := compiler.resolver.ComponentConfig(ctx, location, componentName, cacheDir)

	if err != nil {
		return nil, err
	}

	filename, err := compiler.resolver.PrimaryWeightFile(ctx, location, componentName, cacheDir)

	if err != nil {
		return nil, err
	}

	return map[string]any{
		"kind":     "Block",
		"name":     componentName,
		"category": "model",
		"system": map[string]any{
			"topology": map[string]any{
				"from_safetensors": map[string]any{
					"source":       location.RepoID,
					"file":         filename,
					"architecture": compiler.resolver.ClassName(config, componentName),
					"config":       config,
				},
			},
		},
	}, nil
}

func resolveManifestReferences(root map[string]any) {
	includes := includeMap(root)

	if len(includes) == 0 {
		return
	}

	resolveValueReferences(root, includes)
}

func resolveValueReferences(value any, includes map[string]any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "include" || key == "includes" {
				continue
			}

			typed[key] = resolveValueReferences(child, includes)
		}

		return typed
	case []any:
		for index, child := range typed {
			typed[index] = resolveValueReferences(child, includes)
		}

		return typed
	case string:
		resolved, ok := resolveIncludeReference(typed, includes)

		if ok {
			return resolved
		}

		return typed
	default:
		return value
	}
}

func resolveIncludeReference(reference string, includes map[string]any) (any, bool) {
	includeName, path, ok := strings.Cut(reference, ".")

	if !ok {
		return nil, false
	}

	value, ok := includes[includeName]

	if !ok {
		return nil, false
	}

	return descendManifestPath(value, strings.Split(path, "."))
}

func descendManifestPath(value any, path []string) (any, bool) {
	current := value

	for _, segment := range path {
		document, ok := current.(map[string]any)

		if !ok {
			return nil, false
		}

		current, ok = document[segment]

		if !ok {
			return nil, false
		}
	}

	return current, true
}

/*
NormalizeIncludePath maps template-relative paths for asset FS reads.
*/
func NormalizeIncludePath(name string) string {
	trimmed := strings.TrimSpace(name)

	if trimmed == "" {
		return trimmed
	}

	if strings.HasPrefix(trimmed, "hf://") {
		return trimmed
	}

	if strings.Contains(trimmed, "/") {
		if !strings.HasSuffix(trimmed, ".yml") && !strings.HasSuffix(trimmed, ".yaml") {
			return trimmed + ".yml"
		}

		return trimmed
	}

	if strings.Contains(trimmed, ".") {
		return strings.ReplaceAll(trimmed, ".", "/") + ".yml"
	}

	return path.Join("model", trimmed+".yml")
}
