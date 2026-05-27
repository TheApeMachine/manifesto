package runtime

import (
	"context"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestAxpyOnto(testingObject *testing.T) {
	convey.Convey("Given float32 latent buffers", testingObject, func() {
		target := []float32{1, 2, 3}
		addend := []float32{4, 5, 6}

		updated, err := axpyOnto(context.Background(), nil, target, addend, 0.5)

		convey.So(err, convey.ShouldBeNil)
		convey.So(updated.([]float32), convey.ShouldResemble, []float32{3, 4.5, 6})
	})

	convey.Convey("Given bfloat16 resident latent tensors", testingObject, func() {
		backend := tensor.NewHostBackend()
		defer backend.Close()

		shape, err := tensor.NewShape([]int{3})
		convey.So(err, convey.ShouldBeNil)

		targetValues := []dtype.BF16{
			dtype.NewBfloat16FromFloat32(1),
			dtype.NewBfloat16FromFloat32(2),
			dtype.NewBfloat16FromFloat32(3),
		}

		target, err := backend.Upload(shape, dtype.BFloat16, convert.BFloat16ToBytes(targetValues))
		convey.So(err, convey.ShouldBeNil)

		updated, err := axpyOnto(context.Background(), backend, target, []float32{4, 5, 6}, 0.5)
		convey.So(err, convey.ShouldBeNil)
		defer updated.(tensor.Tensor).Close()

		storageDType, raw, err := backend.Download(updated.(tensor.Tensor))
		convey.So(err, convey.ShouldBeNil)
		convey.So(storageDType, convey.ShouldEqual, dtype.BFloat16)

		values, err := convert.BytesToFloat32(storageDType, raw)
		convey.So(err, convey.ShouldBeNil)
		convey.So(values, convey.ShouldResemble, []float32{3, 4.5, 6})
	})

	convey.Convey("Given resident tensors on an Axpy-capable backend", testingObject, func() {
		backend := &captureAxpyBackend{Backend: tensor.NewHostBackend()}
		target := newDispatchTensor(testingObject, []int{3}, dtype.Float32)
		addend := newDispatchTensor(testingObject, []int{3}, dtype.Float32)

		updated, err := axpyOnto(context.Background(), backend, target, addend, -0.25)

		convey.So(err, convey.ShouldBeNil)
		convey.So(updated, convey.ShouldEqual, target)
		convey.So(backend.called, convey.ShouldBeTrue)
		convey.So(backend.count, convey.ShouldEqual, 3)
		convey.So(backend.alpha, convey.ShouldEqual, float32(-0.25))
		convey.So(backend.format, convey.ShouldEqual, dtype.Float32)
	})
}

func BenchmarkAxpyOnto(benchmark *testing.B) {
	target := make([]float32, 4096)
	addend := make([]float32, 4096)

	for benchmark.Loop() {
		_, _ = axpyOnto(context.Background(), nil, target, addend, 0.25)
	}
}

type captureAxpyBackend struct {
	tensor.Backend
	called bool
	count  int
	alpha  float32
	format dtype.DType
}

func (backend *captureAxpyBackend) Location() tensor.Location {
	return tensor.Metal
}

func (backend *captureAxpyBackend) Axpy(
	y unsafe.Pointer,
	x unsafe.Pointer,
	count int,
	alpha float32,
	format dtype.DType,
) {
	_ = y
	_ = x

	backend.called = true
	backend.count = count
	backend.alpha = alpha
	backend.format = format
}

type dispatchTestTensor struct {
	tensor.Tensor
	location tensor.Location
	format   dtype.DType
	pointer  unsafe.Pointer
	storage  byte
}

func newDispatchTensor(
	testingObject *testing.T,
	dimensions []int,
	format dtype.DType,
) *dispatchTestTensor {
	testingObject.Helper()

	shape, err := tensor.NewShape(dimensions)
	convey.So(err, convey.ShouldBeNil)

	resident, err := tensor.New(shape, format)
	convey.So(err, convey.ShouldBeNil)

	residentTensor := &dispatchTestTensor{
		Tensor:   resident,
		location: tensor.Metal,
		format:   format,
	}
	residentTensor.pointer = unsafe.Pointer(&residentTensor.storage)

	return residentTensor
}

func (resident *dispatchTestTensor) Location() tensor.Location {
	return resident.location
}

func (resident *dispatchTestTensor) DType() dtype.DType {
	return resident.format
}

func (resident *dispatchTestTensor) DispatchPointer() unsafe.Pointer {
	return resident.pointer
}
