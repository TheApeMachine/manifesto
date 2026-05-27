package runtime

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

type axpyDevice interface {
	Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType)
}

type dispatchPointerTensor interface {
	tensor.Tensor
	DispatchPointer() unsafe.Pointer
}

func axpyOnto(
	ctx context.Context,
	memory tensor.Backend,
	target any,
	addend any,
	alpha float32,
) (any, error) {
	resident, ok, err := axpyOntoResident(memory, target, addend, alpha)

	if err != nil || ok {
		return resident, err
	}

	addendVector, err := float32Vector(ctx, addend)

	if err != nil {
		return nil, err
	}

	return axpyOntoHost(memory, target, addendVector, alpha)
}

func axpyOntoResident(
	memory tensor.Backend,
	target any,
	addend any,
	alpha float32,
) (any, bool, error) {
	device, ok := memory.(axpyDevice)

	if !ok {
		return nil, false, nil
	}

	targetTensor, ok := target.(dispatchPointerTensor)

	if !ok {
		return nil, false, nil
	}

	addendTensor, ok := addend.(dispatchPointerTensor)

	if !ok {
		return nil, false, nil
	}

	if targetTensor.Location() != memory.Location() || addendTensor.Location() != memory.Location() {
		return nil, true, fmt.Errorf("math.axpy: tensor location does not match backend")
	}

	if targetTensor.DType() != addendTensor.DType() {
		return nil, true, fmt.Errorf("math.axpy: y dtype %s does not match x dtype %s", targetTensor.DType(), addendTensor.DType())
	}

	if targetTensor.Len() != addendTensor.Len() {
		return nil, true, fmt.Errorf(
			"math.axpy: y length %d does not match x length %d",
			targetTensor.Len(),
			addendTensor.Len(),
		)
	}

	device.Axpy(
		targetTensor.DispatchPointer(),
		addendTensor.DispatchPointer(),
		targetTensor.Len(),
		alpha,
		targetTensor.DType(),
	)

	return targetTensor, true, nil
}

func axpyOntoHost(
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
