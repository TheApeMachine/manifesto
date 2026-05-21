package runtime

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
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
			float32VectorBytes(targetValues),
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
	if storageDType != dtype.Float32 {
		return nil, fmt.Errorf("math.axpy: expected float32 tensor, got %s", storageDType)
	}

	elementCount := len(raw) / 4
	values := make([]float32, elementCount)

	for index := range elementCount {
		values[index] = math.Float32frombits(
			binary.LittleEndian.Uint32(raw[index*4 : index*4+4]),
		)
	}

	return values, nil
}

func float32VectorBytes(values []float32) []byte {
	raw := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(raw[index*4:], math.Float32bits(value))
	}

	return raw
}
