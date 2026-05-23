package runtime

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/diffusion"
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

	width, err := configInt(step.Config, "width")

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	height, err := configInt(step.Config, "height")

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	seed, err := configInt64(step.Config, "seed")

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	latentDownsample, err := configInt(step.Config, "latent_downsample")

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	packedChannels, err := configInt(step.Config, "latent_channels")

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	if !executor.executionDType.IsFloat() {
		return fmt.Errorf("diffusion.prepare_latents: execution dtype is required")
	}

	layout, err := diffusion.ComputeLatentLayout(width, height, latentDownsample, packedChannels)

	if err != nil {
		return fmt.Errorf("diffusion.prepare_latents: %w", err)
	}

	latents, latentIDs, err := uploadPackedLatentsWithIDs(executor.stateMemory, layout, seed, executor.executionDType)

	if err != nil {
		return err
	}

	for name, reference := range step.Out {
		if strings.HasPrefix(reference, "state.") && executor.state != nil {
			value := any(latents)

			if reference == "state.latent_ids" {
				value = latentIDs
			}

			if err := executor.state.SetReference(reference, value); err != nil {
				_ = latents.Close()
				return err
			}

			continue
		}

		values[name] = latents
	}

	if err := executor.bindSchedulerImageSeqLen(schedulerNameFromConfig(step.Config), layout.ImageSeqLen); err != nil {
		return err
	}

	return nil
}

func (executor *Executor) bindSchedulerImageSeqLen(schedulerName string, imageSeqLen int) error {
	if schedulerName == "" {
		return nil
	}

	scheduler, err := executor.scheduler(schedulerName)

	if err != nil {
		return err
	}

	return scheduler.SetImageSeqLen(imageSeqLen)
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

	return executor.bindSchedulerFromLatents(schedulerNameFromConfig(step.Config), latents)
}

func (executor *Executor) bindSchedulerFromLatents(schedulerName string, latents tensor.Tensor) error {
	if schedulerName == "" {
		return nil
	}

	scheduler, err := executor.scheduler(schedulerName)

	if err != nil {
		return err
	}

	dims := latents.Shape().Dims()

	if len(dims) < 2 {
		return fmt.Errorf("scheduler bind latents: expected [batch, seq, channels], got %v", dims)
	}

	return scheduler.SetImageSeqLen(dims[1])
}
