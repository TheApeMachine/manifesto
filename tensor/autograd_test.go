package tensor

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestTape_BackwardAdd(t *testing.T) {
	convey.Convey("Given two scalar tensors and a recorded add op", t, func() {
		shape, _ := NewShape([]int{4})
		left, _ := NewZeroed(shape, dtype.Float64)
		right, _ := NewZeroed(shape, dtype.Float64)
		out, _ := NewZeroed(shape, dtype.Float64)

		_ = left.SetRequiresGrad(true)
		_ = right.SetRequiresGrad(true)
		_ = out.SetRequiresGrad(true)

		leftView, _ := left.Float64Native()
		rightView, _ := right.Float64Native()
		outView, _ := out.Float64Native()

		for index := range leftView {
			leftView[index] = float64(index + 1)
			rightView[index] = float64(10 * (index + 1))
			outView[index] = leftView[index] + rightView[index]
		}

		ctx := context.Background()
		tape := NewTape()

		// Add's backward is simply: grad_left = upstream, grad_right = upstream.
		tape.Record(&SimpleGradFn{
			OpName:    "add",
			InputList: []Tensor{left, right},
			OutTensor: out,
			BackFn: func(ctx context.Context, upstream Tensor) ([]Tensor, error) {
				clone, _ := Contiguous(upstream)
				cloneTwo, _ := Contiguous(upstream)
				return []Tensor{clone, cloneTwo}, nil
			},
		})

		seed, _ := NewZeroed(shape, dtype.Float64)
		seedView, _ := seed.Float64Native()

		for index := range seedView {
			seedView[index] = 1.0
		}

		err := SetHostGrad(out, seed)
		convey.So(err, convey.ShouldBeNil)

		err = tape.Backward(ctx, out)

		convey.Convey("Each input should have a gradient of ones", func() {
			convey.So(err, convey.ShouldBeNil)

			leftGrad, err := left.Grad()
			convey.So(err, convey.ShouldBeNil)
			leftGradView, _ := leftGrad.Float64Native()

			rightGrad, err := right.Grad()
			convey.So(err, convey.ShouldBeNil)
			rightGradView, _ := rightGrad.Float64Native()

			for index := range leftGradView {
				convey.So(leftGradView[index], convey.ShouldEqual, 1.0)
				convey.So(rightGradView[index], convey.ShouldEqual, 1.0)
			}
		})
	})
}

func TestTape_RecordAndClear(t *testing.T) {
	convey.Convey("Given an empty tape", t, func() {
		tape := NewTape()
		convey.So(tape.Length(), convey.ShouldEqual, 0)

		convey.Convey("Recording adds a node", func() {
			tape.Record(&SimpleGradFn{OpName: "noop"})
			convey.So(tape.Length(), convey.ShouldEqual, 1)
		})

		convey.Convey("Clear empties the tape", func() {
			tape.Record(&SimpleGradFn{OpName: "noop"})
			tape.Clear()
			convey.So(tape.Length(), convey.ShouldEqual, 0)
		})
	})
}
