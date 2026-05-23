package compiler

import (
	"context"

	"github.com/theapemachine/manifesto/resolve"
)

func (compiler *Compiler) componentHubVariables(
	ctx context.Context,
	component map[string]any,
	cacheDir string,
) map[string]any {
	source, _ := component["source"].(string)

	if source == "" {
		return nil
	}

	subfolder, _ := component["path"].(string)

	if subfolder == "" {
		if file, ok := component["file"].(string); ok {
			subfolder = safetensorsSubfolder(file)
		}
	}

	if subfolder == "" {
		return nil
	}

	location := resolve.RepoLocation{
		RepoID:   source,
		RepoType: resolve.ModelRepo,
		Revision: "main",
	}

	if revision, ok := component["revision"].(string); ok && revision != "" {
		location.Revision = revision
	}

	hubConfig, err := compiler.resolver.ComponentConfig(ctx, location, subfolder, cacheDir)

	if err != nil {
		return nil
	}

	return componentVariablesFromHubConfig(hubConfig)
}

func componentVariablesFromHubConfig(hubConfig map[string]any) map[string]any {
	variables := normalizeHubConfigMap(hubConfig)

	numLayers := intFromAny(variables["num_hidden_layers"], 0)
	numHeads := intFromAny(variables["num_attention_heads"], 0)
	numKVHeads := intFromAny(variables["num_key_value_heads"], 0)
	headDim := intFromAny(variables["head_dim"], 0)

	variables["q_proj_out"] = numHeads * headDim
	variables["kv_proj_out"] = numKVHeads * headDim

	if _, ok := variables["eps"]; !ok {
		if eps, found := variables["rms_norm_eps"]; found {
			variables["eps"] = eps
		}
	}

	if numLayers > 0 {
		variables["prompt_layer_a"] = numLayers / 4
		variables["prompt_layer_b"] = numLayers / 2
		variables["prompt_layer_c"] = (numLayers * 3) / 4
	}

	return variables
}

func normalizeHubConfigMap(hubConfig map[string]any) map[string]any {
	normalized := make(map[string]any, len(hubConfig))

	for key, value := range hubConfig {
		normalized[key] = normalizeHubConfigValue(value)
	}

	return normalized
}

func normalizeHubConfigValue(value any) any {
	switch typed := value.(type) {
	case float64:
		if typed == float64(int64(typed)) {
			return int(typed)
		}

		return typed
	case []any:
		items := make([]any, len(typed))

		for index, item := range typed {
			items[index] = normalizeHubConfigValue(item)
		}

		return items
	default:
		return value
	}
}

func mergeComponentExpandVariables(
	document map[string]any,
	component map[string]any,
	componentVariables map[string]any,
) map[string]any {
	merged := make(map[string]any)

	if layoutVariables, err := diffusionExpandVariables(document); err == nil {
		merged = mergeConfigMaps(merged, layoutVariables)
	}

	if len(componentVariables) > 0 {
		merged = mergeConfigMaps(merged, componentVariables)
	}

	return merged
}
