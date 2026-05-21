package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestAxpyOnto(testingObject *testing.T) {
	convey.Convey("Given float32 latent buffers", testingObject, func() {
		target := []float32{1, 2, 3}
		addend := []float32{4, 5, 6}

		updated, err := axpyOnto(nil, target, addend, 0.5)

		convey.So(err, convey.ShouldBeNil)
		convey.So(updated.([]float32), convey.ShouldResemble, []float32{3, 4.5, 6})
	})
}

func BenchmarkAxpyOnto(benchmark *testing.B) {
	target := make([]float32, 4096)
	addend := make([]float32, 4096)

	for benchmark.Loop() {
		_, _ = axpyOnto(nil, target, addend, 0.25)
	}
}
