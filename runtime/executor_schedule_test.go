package runtime

import (
	"context"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestExecutorRunEmpiricalMu(testingObject *testing.T) {
	convey.Convey("Given empirical_mu configured with FLUX2 schedule coefficients", testingObject, func() {
		executor := NewExecutor(ExecutorOptions{ExecutionDType: dtype.Float32})
		values := map[string]any{}
		step := ast.Step{
			Op: "math.empirical_mu",
			Config: map[string]any{
				"image_seq_len": 4096,
				"num_steps":     4,
			},
			Out: map[string]string{
				"value": "mu",
			},
		}

		convey.Convey("It should compute the same scalar as the reference schedule", func() {
			err := executor.runEmpiricalMu(step, values)

			convey.So(err, convey.ShouldBeNil)
			convey.So(values["mu"], convey.ShouldAlmostEqual, float32(2.2911799), 1e-6)
		})
	})
}

func TestExecutorRunTimeShift(testingObject *testing.T) {
	convey.Convey("Given raw sigmas and an empirical mu", testingObject, func() {
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name:  "sigmas",
				Type:  "tensor",
				Shape: []any{4},
			},
		})
		convey.So(err, convey.ShouldBeNil)

		executor := NewExecutor(ExecutorOptions{
			State:          state,
			StateMemory:    tensor.NewHostBackend(),
			ExecutionDType: dtype.Float32,
		})
		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "math.linspace",
					Config: map[string]any{
						"start":    1.0,
						"stop":     0.0,
						"count":    4,
						"endpoint": false,
					},
					Out: map[string]string{
						"value": "raw_sigmas",
					},
				},
				{
					Op: "math.empirical_mu",
					Config: map[string]any{
						"image_seq_len": 4096,
						"num_steps":     4,
					},
					Out: map[string]string{
						"value": "mu",
					},
				},
				{
					Op: "math.time_shift",
					In: map[string]string{
						"x":  "raw_sigmas",
						"mu": "mu",
					},
					Config: map[string]any{
						"mode":  "exponential",
						"sigma": 1.0,
					},
					Out: map[string]string{
						"value": "state.sigmas",
					},
				},
			},
		}

		convey.Convey("It should match the shifted FLUX2 sigma schedule", func() {
			err := executor.Run(context.Background(), program, nil, nil)

			convey.So(err, convey.ShouldBeNil)
			sigmas := stateTensorValues(testingObject, state, "sigmas")
			convey.So(sigmas[0], convey.ShouldEqual, float32(1))
			convey.So(sigmas[1], convey.ShouldAlmostEqual, float32(0.96738404), 1e-6)
			convey.So(sigmas[2], convey.ShouldAlmostEqual, float32(0.90814394), 1e-6)
			convey.So(sigmas[3], convey.ShouldAlmostEqual, float32(0.76719993), 1e-6)
		})
	})
}

func TestTimeShiftValue(testingObject *testing.T) {
	convey.Convey("Given the exponential time-shift formula", testingObject, func() {
		mu := float32(2.2911799)
		value := float32(0.75)
		muExp := float32(math.Exp(float64(mu)))
		expected := muExp / (muExp + (1/value - 1))

		convey.Convey("It should apply the reference equation", func() {
			shifted, err := timeShiftValue(value, mu, 1, "exponential")

			convey.So(err, convey.ShouldBeNil)
			convey.So(shifted, convey.ShouldAlmostEqual, expected, 1e-6)
		})
	})
}
