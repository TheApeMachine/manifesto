package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/diffusion"
)

func diffusionLayoutFromDocument(document map[string]any) (diffusion.LatentLayout, int, error) {
	generation, ok := nestedMap(document, "system", "runtime", "generation")

	if !ok {
		return diffusion.LatentLayout{}, 0, fmt.Errorf("diffusion layout: missing system.runtime.generation")
	}

	width := intFromAny(generation["width"], 0)
	height := intFromAny(generation["height"], 0)
	latentDownsample := intFromAny(generation["latent_downsample"], 0)
	packedChannels := intFromAny(generation["latent_channels"], 0)
	contextSeqLen := intFromAny(generation["max_sequence_length"], 1024)

	layout, err := diffusion.ComputeLatentLayout(width, height, latentDownsample, packedChannels)

	if err != nil {
		return diffusion.LatentLayout{}, 0, err
	}

	return layout, contextSeqLen, nil
}

func diffusionExpandVariables(document map[string]any) (map[string]any, error) {
	layout, contextSeqLen, err := diffusionLayoutFromDocument(document)

	if err != nil {
		return nil, err
	}

	return layout.TopologyVariables(contextSeqLen), nil
}

func diffusionRecipeConfig(document map[string]any) (map[string]any, error) {
	layout, contextSeqLen, err := diffusionLayoutFromDocument(document)

	if err != nil {
		return nil, err
	}

	return layout.RecipeConfig(contextSeqLen), nil
}

func mergeConfigMaps(base map[string]any, overlays ...map[string]any) map[string]any {
	merged := make(map[string]any)

	for key, value := range base {
		merged[key] = value
	}

	for _, overlay := range overlays {
		for key, value := range overlay {
			merged[key] = value
		}
	}

	return merged
}

func intFromAny(value any, fallback int) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return fallback
	}
}

func safetensorsSubfolder(filename string) string {
	for index := len(filename) - 1; index >= 0; index-- {
		if filename[index] == '/' {
			return filename[:index]
		}
	}

	return ""
}
