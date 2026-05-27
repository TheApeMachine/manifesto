package runtime

import (
	"context"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestExecutorRunRandomNormal(testingObject *testing.T) {
	convey.Convey("Given a seeded random.normal runtime step", testingObject, func() {
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name:  "latents",
				Type:  "tensor",
				Shape: []any{1, 2},
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
					Op: "random.normal",
					Config: map[string]any{
						"shape": []any{1, 2},
						"seed":  int64(7),
						"dtype": "float32",
					},
					Out: map[string]string{
						"value": "state.latents",
					},
				},
			},
		}

		convey.Convey("It should write deterministic resident state", func() {
			err := executor.Run(context.Background(), program, nil, nil)
			convey.So(err, convey.ShouldBeNil)

			value, ok := state.Get("latents")
			convey.So(ok, convey.ShouldBeTrue)

			stateTensor, ok := value.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(stateTensor.Shape().Dims(), convey.ShouldResemble, []int{1, 2})
			convey.So(stateTensor.DType(), convey.ShouldEqual, dtype.Float32)

			actual, err := stateTensor.Float32Native()
			convey.So(err, convey.ShouldBeNil)

			generator := rand.New(rand.NewSource(7))
			expected := []float32{
				float32(generator.NormFloat64()),
				float32(generator.NormFloat64()),
			}

			convey.So(actual, convey.ShouldResemble, expected)
		})
	})
}

func TestExecutorRunLinspaceScalarBroadcast(testingObject *testing.T) {
	convey.Convey("Given linspace feeding scalar_broadcast", testingObject, func() {
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name:  "sigmas",
				Type:  "tensor",
				Shape: []any{4},
			},
			{
				Name:  "timesteps",
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
						"start": 1.0,
						"stop":  0.0,
						"count": 4,
					},
					Out: map[string]string{
						"value": "state.sigmas",
					},
				},
				{
					Op: "math.scalar_broadcast",
					In: map[string]string{
						"x": "state.sigmas",
					},
					Config: map[string]any{
						"op":     "mul",
						"scalar": 1000.0,
					},
					Out: map[string]string{
						"value": "state.timesteps",
					},
				},
			},
		}

		convey.Convey("It should materialize both schedule tensors", func() {
			err := executor.Run(context.Background(), program, nil, nil)
			convey.So(err, convey.ShouldBeNil)

			sigmas := stateTensorValues(testingObject, state, "sigmas")
			timesteps := stateTensorValues(testingObject, state, "timesteps")

			convey.So(sigmas[0], convey.ShouldEqual, float32(1))
			convey.So(sigmas[1], convey.ShouldAlmostEqual, float32(2.0/3.0), 1e-6)
			convey.So(sigmas[2], convey.ShouldAlmostEqual, float32(1.0/3.0), 1e-6)
			convey.So(sigmas[3], convey.ShouldEqual, float32(0))
			convey.So(timesteps[0], convey.ShouldEqual, float32(1000))
			convey.So(timesteps[1], convey.ShouldAlmostEqual, float32(2000.0/3.0), 1e-3)
			convey.So(timesteps[2], convey.ShouldAlmostEqual, float32(1000.0/3.0), 1e-3)
			convey.So(timesteps[3], convey.ShouldEqual, float32(0))
		})
	})
}

func BenchmarkExecutorRunRandomNormal(benchmark *testing.B) {
	state, err := NewStateStore([]ast.StateDeclaration{
		{
			Name:  "latents",
			Type:  "tensor",
			Shape: []any{1, 4096, 128},
		},
	})

	if err != nil {
		benchmark.Fatal(err)
	}

	executor := NewExecutor(ExecutorOptions{
		State:          state,
		StateMemory:    tensor.NewHostBackend(),
		ExecutionDType: dtype.Float32,
	})
	program := &ast.Program{
		Steps: []ast.Step{
			{
				Op: "random.normal",
				Config: map[string]any{
					"shape": []any{1, 4096, 128},
					"seed":  int64(7),
					"dtype": "float32",
				},
				Out: map[string]string{
					"value": "state.latents",
				},
			},
		},
	}

	for benchmark.Loop() {
		if err := executor.Run(context.Background(), program, nil, nil); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkExecutorRunLinspaceScalarBroadcast(benchmark *testing.B) {
	state, err := NewStateStore([]ast.StateDeclaration{
		{
			Name:  "sigmas",
			Type:  "tensor",
			Shape: []any{4},
		},
		{
			Name:  "timesteps",
			Type:  "tensor",
			Shape: []any{4},
		},
	})

	if err != nil {
		benchmark.Fatal(err)
	}

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
					"start": 1.0,
					"stop":  0.0,
					"count": 4,
				},
				Out: map[string]string{
					"value": "state.sigmas",
				},
			},
			{
				Op: "math.scalar_broadcast",
				In: map[string]string{
					"x": "state.sigmas",
				},
				Config: map[string]any{
					"op":     "mul",
					"scalar": 1000.0,
				},
				Out: map[string]string{
					"value": "state.timesteps",
				},
			},
		},
	}

	for benchmark.Loop() {
		if err := executor.Run(context.Background(), program, nil, nil); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func stateTensorValues(
	testingObject *testing.T,
	state *StateStore,
	name string,
) []float32 {
	testingObject.Helper()

	value, ok := state.Get(name)

	if !ok {
		testingObject.Fatalf("missing state %q", name)
	}

	stateTensor, ok := value.(tensor.Tensor)

	if !ok {
		testingObject.Fatalf("state %q is %T", name, value)
	}

	values, err := stateTensor.Float32Native()

	if err != nil {
		testingObject.Fatalf("state %q values: %v", name, err)
	}

	return append([]float32(nil), values...)
}
