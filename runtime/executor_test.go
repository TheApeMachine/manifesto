package runtime

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestExecutorRunEncode(testingObject *testing.T) {
	convey.Convey("Given a tokenizer encode step with chat template enabled", testingObject, func() {
		host := &encodeCaptureHost{}
		executor := NewExecutor(ExecutorOptions{
			Host: host,
			InitialValues: map[string]any{
				"user_text": "hello",
			},
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "tokenizer.encode",
					In: map[string]string{
						"text": "user_text",
					},
					Out: map[string]string{
						"value": "input_ids",
					},
					Config: map[string]any{
						"tokenizer":           "hf://meta-llama/Llama-3.2-1B-Instruct",
						"tokenizer_file":      "tokenizer/tokenizer.json",
						"apply_chat_template": true,
						"max_length":          512,
						"pad_token_id":        151643,
					},
				},
			},
		}

		err := executor.Run(context.Background(), program, nil, nil)

		convey.So(err, convey.ShouldBeNil)
		convey.So(host.request.ApplyChatTemplate, convey.ShouldBeTrue)
		convey.So(host.request.TokenizerFile, convey.ShouldEqual, "tokenizer/tokenizer.json")
		convey.So(host.request.Text, convey.ShouldEqual, "hello")
		convey.So(host.request.MaxLength, convey.ShouldEqual, 512)
		convey.So(host.request.PadTokenID, convey.ShouldEqual, 151643)
	})
}

func TestExecutorRunWriteImage(testingObject *testing.T) {
	convey.Convey("Given a write image step with image config", testingObject, func() {
		host := &encodeCaptureHost{}
		executor := NewExecutor(ExecutorOptions{
			Host: host,
			InitialValues: map[string]any{
				"image": []float32{1, 0, -1},
			},
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "io.write_image",
					In: map[string]string{
						"image": "image",
					},
					Config: map[string]any{
						"path":     "out.png",
						"width":    1024,
						"height":   512,
						"channels": 3,
						"layout":   "channel_planar",
						"range":    "neg_one_one",
					},
				},
			},
		}

		err := executor.Run(context.Background(), program, nil, nil)

		convey.So(err, convey.ShouldBeNil)
		convey.So(host.imageRequest.Path, convey.ShouldEqual, "out.png")
		convey.So(host.imageRequest.Width, convey.ShouldEqual, 1024)
		convey.So(host.imageRequest.Height, convey.ShouldEqual, 512)
		convey.So(host.imageRequest.Channels, convey.ShouldEqual, 3)
		convey.So(host.imageRequest.Layout, convey.ShouldEqual, "channel_planar")
		convey.So(host.imageRequest.Range, convey.ShouldEqual, "neg_one_one")
	})
}

func TestExecutorRunLoopEach(testingObject *testing.T) {
	convey.Convey("Given a loop_each step with a graph call body", testingObject, func() {
		backend := &graphCaptureBackend{}
		executor := NewExecutor(ExecutorOptions{
			Backend: backend,
			InitialValues: map[string]any{
				"timesteps": []float32{0.5},
			},
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "control.loop_each",
					Loop: &ast.Loop{
						Over: "timesteps",
						As:   "timestep",
					},
					Body: []ast.Step{
						{
							Op:    "graph.call",
							Graph: "transformer",
							In: map[string]string{
								"timestep": "timestep",
							},
							Out: map[string]string{
								"sample": "velocity",
							},
						},
					},
				},
			},
		}

		graphs := map[string]*ast.Graph{
			"transformer": {
				Outputs: map[string]string{"sample": "sample"},
			},
		}

		err := executor.Run(context.Background(), program, graphs, map[string]any{"transformer": "compute"})

		convey.So(err, convey.ShouldBeNil)
		convey.So(backend.request.Inputs["timestep"], convey.ShouldEqual, float32(0.5))
	})
}

func TestSetRuntimeValue(testingObject *testing.T) {
	convey.Convey("Given an existing tensor value", testingObject, func() {
		shape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		first, err := tensor.NewHostBackend().Upload(shape, dtype.Float32, []byte{0, 0, 0x80, 0x3f})
		convey.So(err, convey.ShouldBeNil)

		second, err := tensor.NewHostBackend().Upload(shape, dtype.Float32, []byte{0, 0, 0, 0x40})
		convey.So(err, convey.ShouldBeNil)

		values := map[string]any{"velocity": first}

		convey.Convey("It should close the overwritten tensor", func() {
			setRuntimeValue(values, "velocity", second)

			_, err := first.Float32Native()
			convey.So(err, convey.ShouldEqual, tensor.ErrTensorClosed)
			convey.So(values["velocity"], convey.ShouldEqual, second)
		})
	})
}

func TestSchedulersFromProgram(testingObject *testing.T) {
	convey.Convey("Given a program with a flow match scheduler declaration", testingObject, func() {
		program := &ast.Program{
			Schedulers: map[string]ast.SchedulerDeclaration{
				"scheduler": {
					Type: "flow_match_euler_discrete",
					Config: map[string]any{
						"steps":                50,
						"num_train_timesteps":  1000,
						"shift":                3.0,
						"use_dynamic_shifting": true,
						"time_shift_type":      "exponential",
						"image_seq_len":        4096,
					},
				},
			},
		}

		convey.Convey("It should construct runtime schedulers from declarations", func() {
			schedulers, err := SchedulersFromProgram(program)

			convey.So(err, convey.ShouldBeNil)
			convey.So(schedulers, convey.ShouldContainKey, "scheduler")
			convey.So(schedulers["scheduler"].Timesteps(), convey.ShouldHaveLength, 50)
		})
	})
}

func TestMaterializeStateTensors(testingObject *testing.T) {
	convey.Convey("Given float32 state values and a BF16 runtime dtype", testingObject, func() {
		declarations := []ast.StateDeclaration{
			{
				Name:  "latents",
				Type:  "tensor",
				Shape: []any{1, 2},
				Init:  "gaussian",
				Seed:  int64(0),
			},
		}
		state, err := NewStateStore(declarations)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should store resident tensors in the requested dtype", func() {
			err := MaterializeStateTensors(state, declarations, tensor.NewHostBackend(), dtype.BFloat16)

			convey.So(err, convey.ShouldBeNil)

			value, ok := state.Get("latents")
			convey.So(ok, convey.ShouldBeTrue)

			stateTensor, ok := value.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(stateTensor.DType(), convey.ShouldEqual, dtype.BFloat16)
			convey.So(stateTensor.Shape().Dims(), convey.ShouldResemble, []int{1, 2})
		})
	})
}

func TestProgramSessionRun(testingObject *testing.T) {
	convey.Convey("Given a host-neutral runtime session", testingObject, func() {
		host := &encodeCaptureHost{}
		session, err := NewProgramSession(ProgramSessionOptions{
			Program: &ast.Program{
				Steps: []ast.Step{
					{
						Op: "io.write_image",
						In: map[string]string{
							"image": "image",
						},
						Config: map[string]any{
							"path": "out.png",
						},
					},
				},
			},
			Host: host,
		})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should execute through manifesto runtime", func() {
			err := session.RunWithValues(context.Background(), map[string]any{
				"image": []float32{1, 0, -1},
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(host.imageRequest.Path, convey.ShouldEqual, "out.png")
		})
	})
}

type graphCaptureBackend struct {
	request GraphCallRequest
}

func (backend *graphCaptureBackend) CallGraph(
	ctx context.Context,
	request GraphCallRequest,
) (GraphCallResult, error) {
	_ = ctx
	backend.request = request

	return GraphCallResult{Outputs: map[string]any{"sample": []float32{1}}}, nil
}

type encodeCaptureHost struct {
	request      EncodeRequest
	imageRequest WriteImageRequest
}

func (host *encodeCaptureHost) ReadLine(ctx context.Context) (string, error) {
	_ = ctx

	return "", nil
}

func (host *encodeCaptureHost) EmitToken(ctx context.Context, request EmitTokenRequest) error {
	_ = ctx
	_ = request

	return nil
}

func (host *encodeCaptureHost) WriteImage(ctx context.Context, request WriteImageRequest) error {
	_ = ctx
	host.imageRequest = request

	return nil
}

func (host *encodeCaptureHost) Encode(
	ctx context.Context,
	request EncodeRequest,
) ([]int, error) {
	_ = ctx
	host.request = request

	return []int{1, 2, 3}, nil
}
