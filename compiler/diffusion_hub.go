package compiler

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/resolve"
)

/*
enrichGenerationFromHub fills generation topology fields from component config.json
files on the Hub. Runtime manifests declare source + resolution; checkpoint config
is the source of truth for channel counts and spatial downsampling.
*/
func (compiler *Compiler) enrichGenerationFromHub(
	ctx context.Context,
	document map[string]any,
	cacheDir string,
) error {
	generation, ok := nestedMap(document, "system", "runtime", "generation")

	if !ok {
		return nil
	}

	repoID := diffusionRepoID(document)

	if repoID == "" {
		return nil
	}

	location := resolve.RepoLocation{
		RepoID:   repoID,
		RepoType: resolve.ModelRepo,
		Revision: "main",
	}

	if revision := diffusionRepoRevision(document); revision != "" {
		location.Revision = revision
	}

	transformerConfig, err := compiler.resolver.ComponentConfig(ctx, location, "transformer", cacheDir)

	if err == nil {
		if inChannels, ok := transformerConfig["in_channels"]; ok {
			generation["latent_channels"] = intFromAny(inChannels, 0)
		}
	}

	vaeConfig, err := compiler.resolver.ComponentConfig(ctx, location, "vae", cacheDir)

	if err == nil {
		latentDownsample, downsampleErr := latentDownsampleFromVAEConfig(vaeConfig)

		if downsampleErr == nil {
			generation["latent_downsample"] = latentDownsample
		}
	}

	return nil
}

/*
enrichDocumentFromHub merges Hub component configs into a model block before
reference resolution and compilation.
*/
func (compiler *Compiler) enrichDocumentFromHub(
	ctx context.Context,
	document map[string]any,
	cacheDir string,
) error {
	if err := compiler.enrichGenerationFromHub(ctx, document, cacheDir); err != nil {
		return err
	}

	if err := compiler.enrichSchedulerFromHub(ctx, document, cacheDir); err != nil {
		return err
	}

	return nil
}

func (compiler *Compiler) enrichSchedulerFromHub(
	ctx context.Context,
	document map[string]any,
	cacheDir string,
) error {
	scheduler, ok := nestedMap(document, "system", "runtime", "scheduler")

	if !ok {
		return nil
	}

	preserved := preserveSchedulerRuntimeFields(scheduler)

	source, _ := scheduler["source"].(string)

	if source == "" {
		source = diffusionRepoID(document)
	}

	subfolder, _ := scheduler["path"].(string)

	if subfolder == "" {
		subfolder = "scheduler"
	}

	if source == "" {
		return nil
	}

	location := resolve.RepoLocation{
		RepoID:   source,
		RepoType: resolve.ModelRepo,
		Revision: "main",
	}

	if revision := diffusionRepoRevision(document); revision != "" {
		location.Revision = revision
	}

	hubConfig, err := compiler.resolver.ComponentConfig(ctx, location, subfolder, cacheDir)

	if err != nil {
		return nil
	}

	for key, value := range normalizeHubConfigMap(hubConfig) {
		if strings.HasPrefix(key, "_") {
			continue
		}

		scheduler[key] = value
	}

	for key, value := range preserved {
		scheduler[key] = value
	}

	if schedulerType := schedulerTypeFromHubClass(hubConfig); schedulerType != "" {
		scheduler["type"] = schedulerType
	}

	return nil
}

func preserveSchedulerRuntimeFields(scheduler map[string]any) map[string]any {
	preserved := make(map[string]any)

	for _, key := range []string{
		"num_inference_steps",
		"guidance_scale",
		"source",
		"path",
	} {
		if value, ok := scheduler[key]; ok {
			preserved[key] = value
		}
	}

	return preserved
}

func schedulerTypeFromHubClass(hubConfig map[string]any) string {
	className, _ := hubConfig["_class_name"].(string)

	switch className {
	case "FlowMatchEulerDiscreteScheduler":
		return "flow_match_euler_discrete"
	default:
		return ""
	}
}

func diffusionRepoID(document map[string]any) string {
	runtime, ok := nestedMap(document, "system", "runtime")

	if !ok {
		return ""
	}

	if model, ok := nestedMap(runtime, "model"); ok {
		if source, ok := model["source"].(string); ok && source != "" {
			return source
		}
	}

	topology, ok := nestedMap(document, "system", "topology")

	if !ok {
		return ""
	}

	spec, ok := topology["from_safetensors"].(map[string]any)

	if !ok {
		return ""
	}

	source, _ := spec["source"].(string)

	return source
}

func diffusionRepoRevision(document map[string]any) string {
	runtime, ok := nestedMap(document, "system", "runtime")

	if !ok {
		return ""
	}

	if model, ok := nestedMap(runtime, "model"); ok {
		if revision, ok := model["revision"].(string); ok {
			return revision
		}
	}

	return ""
}

func latentDownsampleFromVAEConfig(vaeConfig map[string]any) (int, error) {
	blockCount, err := vaeBlockOutChannelCount(vaeConfig)

	if err != nil {
		return 0, err
	}

	if blockCount <= 0 {
		return 0, fmt.Errorf("vae config: block_out_channels must not be empty")
	}

	vaeScaleFactor := 1

	for index := 1; index < blockCount; index++ {
		vaeScaleFactor *= 2
	}

	return vaeScaleFactor * 2, nil
}

func vaeBlockOutChannelCount(vaeConfig map[string]any) (int, error) {
	raw, ok := vaeConfig["block_out_channels"]

	if !ok {
		return 0, fmt.Errorf("vae config: missing block_out_channels")
	}

	switch typed := raw.(type) {
	case []any:
		return len(typed), nil
	case []int:
		return len(typed), nil
	case []int64:
		return len(typed), nil
	case []float64:
		return len(typed), nil
	default:
		return 0, fmt.Errorf("vae config: block_out_channels has unsupported type %T", raw)
	}
}
