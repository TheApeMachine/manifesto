package runtime

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/diffusion"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (executor *Executor) runPrepareLatents(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	if executor.stateMemory == nil {
		return fmt.Errorf("diffusion.prepare_latents: state memory backend is required")
	}

	width := intFromConfig(step.Config, "width", 0)
	height := intFromConfig(step.Config, "height", 0)
	seed := int64(intFromConfig(step.Config, "seed", 1337))
	latentDownsample := intFromConfig(step.Config, "latent_downsample", 0)
	packedChannels := intFromConfig(step.Config, "latent_channels", 0)
	storageDType := diffusionStorageDType(step.Config)

	layout, err := diffusion.ComputeLatentLayout(width, height, latentDownsample, packedChannels)

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	latents, err := uploadPackedLatents(executor.stateMemory, layout, seed, storageDType)

	if err != nil {
		return err
	}

	for name, reference := range step.Out {
		if strings.HasPrefix(reference, "state.") && executor.state != nil {
			if err := executor.state.SetReference(reference, latents); err != nil {
				_ = latents.Close()
				return err
			}
		}

		values[name] = latents
	}

	if err := executor.bindSchedulerFromLatents(step.Config["scheduler"], latents); err != nil {
		return err
	}

	return nil
}

func (executor *Executor) runSchedulerBindLatents(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	latentsRef, ok := step.In["latents"]

	if !ok {
		return fmt.Errorf("scheduler.bind_latents: latents input is required")
	}

	latentsValue, err := executor.resolveValue(latentsRef, values)

	if err != nil {
		return err
	}

	latents, ok := latentsValue.(tensor.Tensor)

	if !ok {
		return fmt.Errorf("scheduler.bind_latents: latents is %T, expected tensor.Tensor", latentsValue)
	}

	return executor.bindSchedulerFromLatents(step.Config["scheduler"], latents)
}

func (executor *Executor) bindSchedulerFromLatents(schedulerName any, latents tensor.Tensor) error {
	name, ok := schedulerName.(string)

	if !ok || name == "" {
		return nil
	}

	scheduler, err := executor.scheduler(name)

	if err != nil {
		return err
	}

	dims := latents.Shape().Dims()

	if len(dims) < 2 {
		return fmt.Errorf("scheduler bind latents: expected [batch, seq, channels], got %v", dims)
	}

	scheduler.SetImageSeqLen(dims[1])

	return nil
}

func diffusionStorageDType(config map[string]any) dtype.DType {
	raw, ok := config["dtype"].(string)

	if !ok || raw == "" {
		return dtype.BFloat16
	}

	parsed, err := dtype.Parse(raw)

	if err != nil || !parsed.IsFloat() {
		return dtype.BFloat16
	}

	return parsed
}
