package runtime

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/ast"
)

func (executor *Executor) runEmpiricalMu(
	step ast.Step,
	values map[string]any,
) error {
	imageSeqLen, err := requiredFloat32FromConfig(step.Config, "image_seq_len")

	if err != nil {
		return fmt.Errorf("math.empirical_mu: %w", err)
	}

	numSteps, err := requiredFloat32FromConfig(step.Config, "num_steps")

	if err != nil {
		return fmt.Errorf("math.empirical_mu: %w", err)
	}

	lowStep := float32FromConfig(step.Config, "low_step", 10)
	highStep := float32FromConfig(step.Config, "high_step", 200)
	threshold := float32FromConfig(step.Config, "threshold", 4300)
	lowSlope := float32FromConfig(step.Config, "low_slope", 8.73809524e-05)
	lowIntercept := float32FromConfig(step.Config, "low_intercept", 1.89833333)
	highSlope := float32FromConfig(step.Config, "high_slope", 0.00016927)
	highIntercept := float32FromConfig(step.Config, "high_intercept", 0.45666666)

	mu, err := empiricalMu(
		imageSeqLen,
		numSteps,
		empiricalMuConfig{
			LowStep:       lowStep,
			HighStep:      highStep,
			Threshold:     threshold,
			LowSlope:      lowSlope,
			LowIntercept:  lowIntercept,
			HighSlope:     highSlope,
			HighIntercept: highIntercept,
		},
	)

	if err != nil {
		return fmt.Errorf("math.empirical_mu: %w", err)
	}

	for _, ref := range step.Out {
		setRuntimeValue(values, ref, mu)
	}

	return nil
}

type empiricalMuConfig struct {
	LowStep       float32
	HighStep      float32
	Threshold     float32
	LowSlope      float32
	LowIntercept  float32
	HighSlope     float32
	HighIntercept float32
}

func empiricalMu(imageSeqLen, numSteps float32, config empiricalMuConfig) (float32, error) {
	if config.HighStep == config.LowStep {
		return 0, fmt.Errorf("high_step and low_step must differ")
	}

	highMu := config.HighSlope*imageSeqLen + config.HighIntercept

	if imageSeqLen > config.Threshold {
		return highMu, nil
	}

	lowMu := config.LowSlope*imageSeqLen + config.LowIntercept
	stepSlope := (highMu - lowMu) / (config.HighStep - config.LowStep)
	stepIntercept := highMu - config.HighStep*stepSlope

	return stepSlope*numSteps + stepIntercept, nil
}

func (executor *Executor) runTimeShift(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	sourceRef, ok := step.In["x"]

	if !ok {
		return fmt.Errorf("math.time_shift: x input is required")
	}

	muRef, ok := step.In["mu"]

	if !ok {
		return fmt.Errorf("math.time_shift: mu input is required")
	}

	sourceValue, err := executor.resolveValue(sourceRef, values)

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	source, err := float32Vector(ctx, sourceValue)

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	muValue, err := executor.resolveValue(muRef, values)

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	mu, err := runtimeScalarFloat32(muValue)

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	mode, _ := step.Config["mode"].(string)
	sigma := float32FromConfig(step.Config, "sigma", 1)
	output, err := timeShiftValues(source, mu, sigma, mode)

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	shape, err := shapeFromRuntimeValue(sourceValue, len(source))

	if err != nil {
		return fmt.Errorf("math.time_shift: %w", err)
	}

	return executor.writeFloat32VectorOutputs(values, step.Out, float32VectorOutput{
		shape:  shape,
		values: output,
		format: sourceFloatDType(sourceValue),
	})
}

func timeShiftValues(source []float32, mu, sigma float32, mode string) ([]float32, error) {
	output := make([]float32, len(source))

	for index, value := range source {
		shifted, err := timeShiftValue(value, mu, sigma, mode)

		if err != nil {
			return nil, err
		}

		output[index] = shifted
	}

	return output, nil
}

func timeShiftValue(value, mu, sigma float32, mode string) (float32, error) {
	if value < 0 || value > 1 {
		return 0, fmt.Errorf("value %g outside [0,1]", value)
	}

	if value == 0 {
		return 0, nil
	}

	switch mode {
	case "", "exponential":
		muExp := float32(math.Exp(float64(mu)))
		denominator := muExp + float32(math.Pow(float64(1/value-1), float64(sigma)))
		return muExp / denominator, nil
	case "linear":
		denominator := mu + float32(math.Pow(float64(1/value-1), float64(sigma)))
		return mu / denominator, nil
	default:
		return 0, fmt.Errorf("unsupported mode %q", mode)
	}
}

func (executor *Executor) runSchedulerDelta(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	scheduleRef, ok := step.In["schedule"]

	if !ok {
		return fmt.Errorf("math.scheduler_delta: schedule input is required")
	}

	stepIndexRef, ok := step.In["step_index"]

	if !ok {
		return fmt.Errorf("math.scheduler_delta: step_index input is required")
	}

	scheduleValue, err := executor.resolveValue(scheduleRef, values)

	if err != nil {
		return fmt.Errorf("math.scheduler_delta: %w", err)
	}

	schedule, err := float32Vector(ctx, scheduleValue)

	if err != nil {
		return fmt.Errorf("math.scheduler_delta: %w", err)
	}

	stepIndexValue, err := executor.resolveValue(stepIndexRef, values)

	if err != nil {
		return fmt.Errorf("math.scheduler_delta: %w", err)
	}

	stepIndex, err := intFromRuntimeScalar(stepIndexValue)

	if err != nil {
		return fmt.Errorf("math.scheduler_delta: %w", err)
	}

	delta, err := schedulerDelta(schedule, stepIndex, float32FromConfig(step.Config, "terminal", 0))

	if err != nil {
		return fmt.Errorf("math.scheduler_delta: %w", err)
	}

	for _, ref := range step.Out {
		setRuntimeValue(values, ref, delta)
	}

	return nil
}

func schedulerDelta(schedule []float32, stepIndex int, terminal float32) (float32, error) {
	if len(schedule) == 0 {
		return 0, fmt.Errorf("schedule is empty")
	}

	if stepIndex < 0 || stepIndex >= len(schedule) {
		return 0, fmt.Errorf("step index %d outside schedule length %d", stepIndex, len(schedule))
	}

	current := schedule[stepIndex]
	next := terminal

	if stepIndex+1 < len(schedule) {
		next = schedule[stepIndex+1]
	}

	return next - current, nil
}

func intFromRuntimeScalar(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float32:
		converted := int(typed)

		if float32(converted) != typed {
			return 0, fmt.Errorf("non-integral scalar %v", typed)
		}

		return converted, nil
	case float64:
		converted := int(typed)

		if float64(converted) != typed {
			return 0, fmt.Errorf("non-integral scalar %v", typed)
		}

		return converted, nil
	default:
		return 0, fmt.Errorf("unsupported scalar type %T", value)
	}
}

func runtimeScalarFloat32(value any) (float32, error) {
	vector, err := float32VectorValue(value)

	if err != nil {
		return 0, err
	}

	if len(vector) != 1 {
		return 0, fmt.Errorf("expected scalar, got vector length %d", len(vector))
	}

	return vector[0], nil
}
