package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestAxpyOnto(testingObject *testing.T) {
	convey.Convey("Given float32 latent buffers", testingObject, func() {
		target := []float32{1, 2, 3}
		addend := []float32{4, 5, 6}

		updated, err := axpyOnto(nil, target, addend, 0.5)

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

		updated, err := axpyOnto(backend, target, []float32{4, 5, 6}, 0.5)
		convey.So(err, convey.ShouldBeNil)
		defer updated.(tensor.Tensor).Close()

		storageDType, raw, err := backend.Download(updated.(tensor.Tensor))
		convey.So(err, convey.ShouldBeNil)
		convey.So(storageDType, convey.ShouldEqual, dtype.BFloat16)

		values, err := convert.BytesToFloat32(storageDType, raw)
		convey.So(err, convey.ShouldBeNil)
		convey.So(values, convey.ShouldResemble, []float32{3, 4.5, 6})
	})
}

func BenchmarkAxpyOnto(benchmark *testing.B) {
	target := make([]float32, 4096)
	addend := make([]float32, 4096)

	for benchmark.Loop() {
		_, _ = axpyOnto(nil, target, addend, 0.25)
	}
}
