package runtime

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func axpyOnto(
	memory tensor.Backend,
	target any,
	addend []float32,
	alpha float32,
) (any, error) {
	if len(addend) == 0 {
		return nil, fmt.Errorf("math.axpy: addend vector is empty")
	}

	switch typedTarget := target.(type) {
	case []float32:
		if len(typedTarget) != len(addend) {
			return nil, fmt.Errorf(
				"math.axpy: y length %d does not match x length %d",
				len(typedTarget),
				len(addend),
			)
		}

		for index := range typedTarget {
			typedTarget[index] += alpha * addend[index]
		}

		return typedTarget, nil
	case tensor.Tensor:
		if memory == nil {
			return nil, fmt.Errorf("math.axpy: tensor backend is required")
		}

		storageDType, raw, err := memory.Download(typedTarget)

		if err != nil {
			return nil, err
		}

		targetValues, err := bytesToFloat32Vector(storageDType, raw)

		if err != nil {
			return nil, err
		}

		if len(targetValues) != len(addend) {
			return nil, fmt.Errorf(
				"math.axpy: y length %d does not match x length %d",
				len(targetValues),
				len(addend),
			)
		}

		for index := range targetValues {
			targetValues[index] += alpha * addend[index]
		}

		updated, err := memory.Upload(
			typedTarget.Shape(),
			storageDType,
			float32VectorBytes(storageDType, targetValues),
		)

		if err != nil {
			return nil, err
		}

		_ = typedTarget.Close()

		return updated, nil
	default:
		return nil, fmt.Errorf("math.axpy: unsupported y type %T", target)
	}
}

func bytesToFloat32Vector(storageDType dtype.DType, raw []byte) ([]float32, error) {
	return convert.BytesToFloat32(storageDType, raw)
}

func float32VectorBytes(storageDType dtype.DType, values []float32) []byte {
	switch storageDType {
	case dtype.Float32:
		return convert.Float32ToBytes(values)
	case dtype.Float16:
		converted := make([]dtype.F16, len(values))

		for index, value := range values {
			converted[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(converted)
	case dtype.BFloat16:
		converted := make([]dtype.BF16, len(values))

		for index, value := range values {
			converted[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(converted)
	default:
		return convert.Float32ToBytes(values)
	}
}
