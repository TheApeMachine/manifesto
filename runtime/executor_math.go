package runtime

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type float32VectorOutput struct {
	shape  tensor.Shape
	values []float32
	format dtype.DType
}

func (executor *Executor) runRandomNormal(
	step ast.Step,
	values map[string]any,
) error {
	shape, err := requiredShapeFromConfig(step.Config, "shape")

	if err != nil {
		return fmt.Errorf("random.normal: %w", err)
	}

	seed, err := requiredInt64FromConfig(step.Config, "seed")

	if err != nil {
		return fmt.Errorf("random.normal: %w", err)
	}

	format, err := floatDTypeFromConfig(step.Config, executor.defaultFloatDType())

	if err != nil {
		return fmt.Errorf("random.normal: %w", err)
	}

	generator := rand.New(rand.NewSource(seed))
	output := make([]float32, shape.Len())

	for outputIndex := range output {
		output[outputIndex] = float32(generator.NormFloat64())
	}

	return executor.writeFloat32VectorOutputs(values, step.Out, float32VectorOutput{
		shape:  shape,
		values: output,
		format: format,
	})
}

func (executor *Executor) runLinspace(
	step ast.Step,
	values map[string]any,
) error {
	start, err := requiredFloat32FromConfig(step.Config, "start")

	if err != nil {
		return fmt.Errorf("math.linspace: %w", err)
	}

	stop, err := requiredFloat32FromConfig(step.Config, "stop")

	if err != nil {
		return fmt.Errorf("math.linspace: %w", err)
	}

	count, err := requiredIntFromConfig(step.Config, "count")

	if err != nil {
		return fmt.Errorf("math.linspace: %w", err)
	}

	if count <= 0 {
		return fmt.Errorf("math.linspace: count must be positive")
	}

	output := make([]float32, count)

	if count == 1 {
		output[0] = start
		return executor.writeLinspaceOutput(values, step, output)
	}

	stepSize := (stop - start) / float32(count-1)

	for outputIndex := range output {
		output[outputIndex] = start + float32(outputIndex)*stepSize
	}

	output[0] = start
	output[count-1] = stop

	return executor.writeLinspaceOutput(values, step, output)
}

func (executor *Executor) writeLinspaceOutput(
	values map[string]any,
	step ast.Step,
	output []float32,
) error {
	count := len(output)
	shape, err := tensor.NewShape([]int{count})

	if err != nil {
		return err
	}

	format, err := floatDTypeFromConfig(step.Config, dtype.Float32)

	if err != nil {
		return fmt.Errorf("math.linspace: %w", err)
	}

	return executor.writeFloat32VectorOutputs(values, step.Out, float32VectorOutput{
		shape:  shape,
		values: output,
		format: format,
	})
}

func (executor *Executor) runScalarBroadcast(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	sourceRef, ok := step.In["x"]

	if !ok {
		return fmt.Errorf("math.scalar_broadcast: x input is required")
	}

	source, err := executor.resolveValue(sourceRef, values)

	if err != nil {
		return fmt.Errorf("math.scalar_broadcast: %w", err)
	}

	sourceValues, err := float32Vector(ctx, source)

	if err != nil {
		return fmt.Errorf("math.scalar_broadcast: %w", err)
	}

	scalar, err := requiredFloat32FromConfig(step.Config, "scalar")

	if err != nil {
		return fmt.Errorf("math.scalar_broadcast: %w", err)
	}

	operation, ok := step.Config["op"].(string)

	if !ok || operation == "" {
		return fmt.Errorf("math.scalar_broadcast: op is required")
	}

	output, err := applyScalarBroadcast(sourceValues, scalar, operation)

	if err != nil {
		return err
	}

	shape, err := shapeFromRuntimeValue(source, len(sourceValues))

	if err != nil {
		return fmt.Errorf("math.scalar_broadcast: %w", err)
	}

	format, err := floatDTypeFromConfig(step.Config, sourceFloatDType(source))

	if err != nil {
		return fmt.Errorf("math.scalar_broadcast: %w", err)
	}

	return executor.writeFloat32VectorOutputs(values, step.Out, float32VectorOutput{
		shape:  shape,
		values: output,
		format: format,
	})
}

func applyScalarBroadcast(
	source []float32,
	scalar float32,
	operation string,
) ([]float32, error) {
	output := make([]float32, len(source))

	switch operation {
	case "add":
		for outputIndex, value := range source {
			output[outputIndex] = value + scalar
		}

		return output, nil
	case "sub":
		for outputIndex, value := range source {
			output[outputIndex] = value - scalar
		}

		return output, nil
	case "mul":
		for outputIndex, value := range source {
			output[outputIndex] = value * scalar
		}

		return output, nil
	case "div":
		if scalar == 0 {
			return nil, fmt.Errorf("math.scalar_broadcast: scalar divisor must be nonzero")
		}

		for outputIndex, value := range source {
			output[outputIndex] = value / scalar
		}

		return output, nil
	default:
		return nil, fmt.Errorf("math.scalar_broadcast: unsupported op %q", operation)
	}
}

func (executor *Executor) writeFloat32VectorOutputs(
	values map[string]any,
	outputs map[string]string,
	payload float32VectorOutput,
) error {
	if len(outputs) == 0 {
		return fmt.Errorf("runtime vector output requires at least one target")
	}

	for _, reference := range outputs {
		if err := executor.writeFloat32VectorOutput(values, reference, payload); err != nil {
			return err
		}
	}

	return nil
}

func (executor *Executor) writeFloat32VectorOutput(
	values map[string]any,
	reference string,
	payload float32VectorOutput,
) error {
	copied := append([]float32(nil), payload.values...)

	if !strings.HasPrefix(reference, "state.") {
		setRuntimeValue(values, reference, copied)
		return nil
	}

	if executor.state == nil {
		return fmt.Errorf("state output %q requires a state store", reference)
	}

	var output any = copied

	if executor.stateMemory != nil {
		tensorValue, err := executor.stateMemory.Upload(
			payload.shape,
			payload.format,
			Float32AsDTypeBytes(copied, payload.format),
		)

		if err != nil {
			return err
		}

		output = tensorValue
	}

	if err := executor.state.SetReference(reference, output); err != nil {
		closeRuntimeValue(output)
		return err
	}

	return nil
}

func (executor *Executor) defaultFloatDType() dtype.DType {
	if executor.executionDType.IsFloat() {
		return executor.executionDType
	}

	return dtype.Float32
}

func sourceFloatDType(value any) dtype.DType {
	sourceTensor, ok := value.(tensor.Tensor)

	if ok && sourceTensor.DType().IsFloat() {
		return sourceTensor.DType()
	}

	return dtype.Float32
}

func shapeFromRuntimeValue(value any, length int) (tensor.Shape, error) {
	sourceTensor, ok := value.(tensor.Tensor)

	if ok {
		return sourceTensor.Shape(), nil
	}

	return tensor.NewShape([]int{length})
}

func requiredShapeFromConfig(config map[string]any, key string) (tensor.Shape, error) {
	raw, ok := config[key]

	if !ok {
		return tensor.Shape{}, fmt.Errorf("%s is required", key)
	}

	dimensions, err := dimensionsFromAny(raw)

	if err != nil {
		return tensor.Shape{}, fmt.Errorf("%s: %w", key, err)
	}

	return tensor.NewShape(dimensions)
}

func dimensionsFromAny(raw any) ([]int, error) {
	switch typed := raw.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		dimensions := make([]int, 0, len(typed))

		for _, value := range typed {
			dimension, err := requiredIntFromAny(value)

			if err != nil {
				return nil, err
			}

			dimensions = append(dimensions, dimension)
		}

		return dimensions, nil
	default:
		return nil, fmt.Errorf("expected []int or []any, got %T", raw)
	}
}

func floatDTypeFromConfig(config map[string]any, fallback dtype.DType) (dtype.DType, error) {
	raw, ok := config["dtype"]

	if !ok {
		return fallback, nil
	}

	text, ok := raw.(string)

	if !ok {
		return dtype.Invalid, fmt.Errorf("dtype must be a string")
	}

	format, err := dtype.Parse(text)

	if err != nil {
		return dtype.Invalid, err
	}

	if !format.IsFloat() {
		return dtype.Invalid, fmt.Errorf("dtype %s is not floating point", format)
	}

	return format, nil
}

func requiredIntFromConfig(config map[string]any, key string) (int, error) {
	raw, ok := config[key]

	if !ok {
		return 0, fmt.Errorf("%s is required", key)
	}

	return requiredIntFromAny(raw)
}

func requiredInt64FromConfig(config map[string]any, key string) (int64, error) {
	raw, ok := config[key]

	if !ok {
		return 0, fmt.Errorf("%s is required", key)
	}

	switch typed := raw.(type) {
	case int:
		return int64(typed), nil
	case int64:
		return typed, nil
	case float64:
		return int64(typed), nil
	case string:
		value, err := strconv.ParseInt(typed, 10, 64)

		if err != nil {
			return 0, err
		}

		return value, nil
	default:
		return 0, fmt.Errorf("%s must be an integer, got %T", key, raw)
	}
}

func requiredIntFromAny(raw any) (int, error) {
	switch typed := raw.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		value, err := strconv.Atoi(typed)

		if err != nil {
			return 0, err
		}

		return value, nil
	default:
		return 0, fmt.Errorf("expected integer, got %T", raw)
	}
}

func requiredFloat32FromConfig(config map[string]any, key string) (float32, error) {
	raw, ok := config[key]

	if !ok {
		return 0, fmt.Errorf("%s is required", key)
	}

	switch typed := raw.(type) {
	case float32:
		return typed, nil
	case float64:
		return float32(typed), nil
	case int:
		return float32(typed), nil
	case int64:
		return float32(typed), nil
	case string:
		value, err := strconv.ParseFloat(typed, 32)

		if err != nil {
			return 0, err
		}

		return float32(value), nil
	default:
		return 0, fmt.Errorf("%s must be numeric, got %T", key, raw)
	}
}
