package runtime

import (
	"fmt"
	"math"
)

/*
FlowMatchEulerDiscrete implements the FLUX-style flow-match Euler scheduler.
*/
type FlowMatchEulerDiscrete struct {
	Steps             int
	NumTrainTimesteps int
	Shift             float64
	UseDynamicShift   bool
	TimeShiftType     string
	ImageSeqLen       int
	sigmas            []float32
	timesteps         []float32
}

/*
NewFlowMatchEulerDiscrete constructs a scheduler from a program declaration.
*/
func NewFlowMatchEulerDiscrete(declaration SchedulerConfig) (*FlowMatchEulerDiscrete, error) {
	if declaration.Steps <= 0 {
		return nil, fmt.Errorf("scheduler: steps must be positive")
	}

	if declaration.NumTrainTimesteps <= 0 {
		return nil, fmt.Errorf("scheduler: num_train_timesteps must be positive")
	}

	if declaration.Shift == 0 {
		return nil, fmt.Errorf("scheduler: shift is required")
	}

	if declaration.TimeShiftType == "" {
		return nil, fmt.Errorf("scheduler: time_shift_type is required")
	}

	if !declaration.UseDynamicShift && declaration.ImageSeqLen <= 0 {
		return nil, fmt.Errorf("scheduler: image_seq_len is required when dynamic shifting is disabled")
	}

	return &FlowMatchEulerDiscrete{
		Steps:             declaration.Steps,
		NumTrainTimesteps: declaration.NumTrainTimesteps,
		Shift:             declaration.Shift,
		UseDynamicShift:   declaration.UseDynamicShift,
		TimeShiftType:     declaration.TimeShiftType,
		ImageSeqLen:       declaration.ImageSeqLen,
	}, nil
}

func schedulerConfigFromHub(hubConfig map[string]any, variables map[string]any) (SchedulerConfig, error) {
	steps := intFromAny(variables["num_inference_steps"], 0)

	if steps <= 0 {
		steps = intFromAny(hubConfig["num_inference_steps"], 0)
	}

	if steps <= 0 {
		return SchedulerConfig{}, fmt.Errorf("scheduler: num_inference_steps is required")
	}

	numTrainTimesteps := intFromAny(hubConfig["num_train_timesteps"], 0)

	if numTrainTimesteps <= 0 {
		return SchedulerConfig{}, fmt.Errorf("scheduler: num_train_timesteps is required")
	}

	shift := float64FromAny(hubConfig["shift"], 0)

	if shift == 0 {
		return SchedulerConfig{}, fmt.Errorf("scheduler: shift is required")
	}

	useDynamicShift, ok := hubConfig["use_dynamic_shifting"].(bool)

	if !ok {
		return SchedulerConfig{}, fmt.Errorf("scheduler: use_dynamic_shifting is required")
	}

	timeShiftType, ok := hubConfig["time_shift_type"].(string)

	if !ok || timeShiftType == "" {
		return SchedulerConfig{}, fmt.Errorf("scheduler: time_shift_type is required")
	}

	config := SchedulerConfig{
		Steps:             steps,
		NumTrainTimesteps: numTrainTimesteps,
		Shift:             shift,
		UseDynamicShift:   useDynamicShift,
		TimeShiftType:     timeShiftType,
	}

	if !useDynamicShift {
		imageSeqLen := intFromAny(hubConfig["image_seq_len"], 0)

		if imageSeqLen <= 0 {
			return SchedulerConfig{}, fmt.Errorf("scheduler: image_seq_len is required")
		}

		config.ImageSeqLen = imageSeqLen
	}

	return config, nil
}

func schedulerConfigFromDeclaration(config map[string]any) (SchedulerConfig, error) {
	if config == nil {
		return SchedulerConfig{}, fmt.Errorf("scheduler declaration config is required")
	}

	declaration := SchedulerConfig{
		Steps:             intFromAny(config["steps"], 0),
		NumTrainTimesteps: intFromAny(config["num_train_timesteps"], 0),
		Shift:             float64FromAny(config["shift"], 0),
		UseDynamicShift:   boolFromAny(config["use_dynamic_shifting"]),
		TimeShiftType:     stringFromAny(config["time_shift_type"], ""),
		ImageSeqLen:       intFromAny(config["image_seq_len"], 0),
	}

	return declaration, nil
}

/*
SchedulerConfig is the host-neutral scheduler configuration.
*/
type SchedulerConfig struct {
	Steps             int
	NumTrainTimesteps int
	Shift             float64
	UseDynamicShift   bool
	TimeShiftType     string
	ImageSeqLen       int
}

/*
Timesteps returns the inference timestep schedule.
*/
func (scheduler *FlowMatchEulerDiscrete) Timesteps() []float32 {
	if len(scheduler.timesteps) == scheduler.Steps {
		return append([]float32(nil), scheduler.timesteps...)
	}

	sigmas := scheduler.inferenceSigmas()
	timesteps := make([]float32, scheduler.Steps)

	for index, sigma := range sigmas[:scheduler.Steps] {
		timesteps[index] = sigma * float32(scheduler.NumTrainTimesteps)
	}

	scheduler.sigmas = sigmas
	scheduler.timesteps = timesteps

	return append([]float32(nil), timesteps...)
}

func (scheduler *FlowMatchEulerDiscrete) inferenceSigmas() []float32 {
	sigmas := make([]float32, scheduler.Steps+1)

	for index := range scheduler.Steps {
		sigma := 1.0 - (1.0-float64(1)/float64(scheduler.Steps))*float64(index)/float64(scheduler.Steps-1)

		if scheduler.UseDynamicShift {
			mu := scheduler.empiricalMu()
			sigma = scheduler.timeShift(mu, sigma)
		} else {
			sigma = scheduler.Shift * sigma / (1 + (scheduler.Shift-1)*sigma)
		}

		sigmas[index] = float32(sigma)
	}

	return sigmas
}

func (scheduler *FlowMatchEulerDiscrete) empiricalMu() float64 {
	a1, b1 := 8.73809524e-05, 1.89833333
	a2, b2 := 0.00016927, 0.45666666
	imageSeqLen := float64(scheduler.ImageSeqLen)

	if scheduler.ImageSeqLen > 4300 {
		return a2*imageSeqLen + b2
	}

	m200 := a2*imageSeqLen + b2
	m10 := a1*imageSeqLen + b1
	slope := (m200 - m10) / 190
	intercept := m200 - 200*slope

	return slope*float64(scheduler.Steps) + intercept
}

/*
SetImageSeqLen updates dynamic-shift scheduling from packed latent token count.
*/
func (scheduler *FlowMatchEulerDiscrete) SetImageSeqLen(imageSeqLen int) error {
	if imageSeqLen <= 0 {
		return fmt.Errorf("scheduler: image_seq_len must be positive, got %d", imageSeqLen)
	}

	scheduler.ImageSeqLen = imageSeqLen
	scheduler.sigmas = nil
	scheduler.timesteps = nil

	return nil
}

func (scheduler *FlowMatchEulerDiscrete) timeShift(mu float64, sigma float64) float64 {
	if scheduler.TimeShiftType == "linear" {
		return mu / (mu + (1/sigma - 1))
	}

	expMu := math.Exp(mu)
	return expMu / (expMu + (1/sigma - 1))
}

func (scheduler *FlowMatchEulerDiscrete) Delta(timestep float32) float32 {
	if len(scheduler.sigmas) != scheduler.Steps+1 {
		scheduler.Timesteps()
	}

	sigmaIndex := scheduler.sigmaIndex(timestep)

	return scheduler.sigmas[sigmaIndex+1] - scheduler.sigmas[sigmaIndex]
}

func (scheduler *FlowMatchEulerDiscrete) DeltaForStepIndex(stepIndex int) (float32, error) {
	if len(scheduler.sigmas) != scheduler.Steps+1 {
		scheduler.Timesteps()
	}

	if stepIndex < 0 || stepIndex >= scheduler.Steps {
		return 0, fmt.Errorf("scheduler: step_index %d out of range [0, %d)", stepIndex, scheduler.Steps)
	}

	return scheduler.sigmas[stepIndex+1] - scheduler.sigmas[stepIndex], nil
}

func (scheduler *FlowMatchEulerDiscrete) sigmaIndex(timestep float32) int {
	timesteps := scheduler.Timesteps()
	bestIndex := 0
	bestDistance := float32(math.MaxFloat32)

	for index, candidate := range timesteps {
		distance := float32(math.Abs(float64(candidate - timestep)))

		if distance < bestDistance {
			bestDistance = distance
			bestIndex = index
		}
	}

	return bestIndex
}
