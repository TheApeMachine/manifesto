package runtime

import (
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
	steps := declaration.Steps

	if steps <= 0 {
		steps = 28
	}

	trainSteps := declaration.NumTrainTimesteps

	if trainSteps <= 0 {
		trainSteps = 1000
	}

	shift := declaration.Shift

	if shift == 0 {
		shift = 1
	}

	timeShiftType := declaration.TimeShiftType

	if timeShiftType == "" {
		timeShiftType = "exponential"
	}

	imageSeqLen := declaration.ImageSeqLen

	if imageSeqLen <= 0 {
		imageSeqLen = 4096
	}

	return &FlowMatchEulerDiscrete{
		Steps:             steps,
		NumTrainTimesteps: trainSteps,
		Shift:             shift,
		UseDynamicShift:   declaration.UseDynamicShift,
		TimeShiftType:     timeShiftType,
		ImageSeqLen:       imageSeqLen,
	}, nil
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
func (scheduler *FlowMatchEulerDiscrete) SetImageSeqLen(imageSeqLen int) {
	if imageSeqLen <= 0 {
		return
	}

	scheduler.ImageSeqLen = imageSeqLen
	scheduler.sigmas = nil
	scheduler.timesteps = nil
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

func (scheduler *FlowMatchEulerDiscrete) DeltaForStepIndex(stepIndex int) float32 {
	if len(scheduler.sigmas) != scheduler.Steps+1 {
		scheduler.Timesteps()
	}

	if stepIndex < 0 || stepIndex >= scheduler.Steps {
		return 0
	}

	return scheduler.sigmas[stepIndex+1] - scheduler.sigmas[stepIndex]
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
