package runtime

import (
	"fmt"
)

/*
FlowMatchEulerDiscrete implements the FLUX-style flow-match Euler scheduler.
*/
type FlowMatchEulerDiscrete struct {
	Steps             int
	NumTrainTimesteps int
}

/*
NewFlowMatchEulerDiscrete constructs a scheduler from a program declaration.
*/
func NewFlowMatchEulerDiscrete(declaration SchedulerConfig) (*FlowMatchEulerDiscrete, error) {
	steps := declaration.Steps

	if steps <= 0 {
		steps = 28
	}

	trainSteps := declaration.NumTrainTimesteps

	if trainSteps <= 0 {
		trainSteps = 1000
	}

	return &FlowMatchEulerDiscrete{
		Steps:             steps,
		NumTrainTimesteps: trainSteps,
	}, nil
}

/*
SchedulerConfig is the host-neutral scheduler configuration.
*/
type SchedulerConfig struct {
	Steps             int
	NumTrainTimesteps int
}

/*
Timesteps returns the inference timestep schedule.
*/
func (scheduler *FlowMatchEulerDiscrete) Timesteps() []float32 {
	timesteps := make([]float32, scheduler.Steps)
	stepSize := float32(scheduler.NumTrainTimesteps) / float32(scheduler.Steps)

	for index := range timesteps {
		timesteps[index] = float32(scheduler.NumTrainTimesteps) - float32(index)*stepSize
	}

	return timesteps
}

/*
Step applies one Euler update to latent values.
*/
func (scheduler *FlowMatchEulerDiscrete) Step(
	latents []float32,
	velocity []float32,
	timestep float32,
) ([]float32, error) {
	if len(latents) != len(velocity) {
		return nil, fmt.Errorf("scheduler step: latents/velocity length mismatch")
	}

	stepSize := float32(scheduler.NumTrainTimesteps) / float32(scheduler.Steps)
	nextTimestep := timestep - stepSize

	if nextTimestep < 0 {
		nextTimestep = 0
	}

	delta := (timestep - nextTimestep) / float32(scheduler.NumTrainTimesteps)
	updated := make([]float32, len(latents))

	for index := range latents {
		updated[index] = latents[index] - delta*velocity[index]
	}

	return updated, nil
}
