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
		convey.So(host.request.ChatContinuation, convey.ShouldBeFalse)
		convey.So(host.request.MaxLength, convey.ShouldEqual, 512)
		convey.So(host.request.PadTokenID, convey.ShouldEqual, 151643)
	})
}

func TestExecutorRunEncodeAppendToHistory(testingObject *testing.T) {
	convey.Convey("Given a tokenizer encode step appending to existing chat tokens", testingObject, func() {
		host := &encodeCaptureHost{tokens: []int{20, 21}}
		backend := &graphCaptureBackend{}
		executor := NewExecutor(ExecutorOptions{
			Backend: backend,
			Host:    host,
			InitialValues: map[string]any{
				"input_ids": []int{10, 11},
				"user_text": "what?",
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
						"apply_chat_template": true,
						"append_to":           "input_ids",
					},
				},
				{
					Op:    "graph.call",
					Graph: "model",
					In: map[string]string{
						"input_ids": "input_ids",
					},
				},
			},
		}

		graphs := map[string]*ast.Graph{
			"model": {},
		}

		err := executor.Run(context.Background(), program, graphs, map[string]any{"model": "compute"})

		convey.So(err, convey.ShouldBeNil)
		convey.So(host.request.ChatContinuation, convey.ShouldBeTrue)
		convey.So(backend.request.Inputs["input_ids"], convey.ShouldResemble, []int{10, 11, 20, 21})
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

func TestExecutorRunGraphCallWritesStateOutput(testingObject *testing.T) {
	convey.Convey("Given a graph call that outputs to runtime state", testingObject, func() {
		cache := &PagedTensorState{
			Shape:     []int{2, 4, 8},
			PageSize:  4,
			PageCount: 2,
		}
		backend := &graphOutputBackend{
			outputs: map[string]any{
				"cache": cache,
			},
		}
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name: "cache",
				Type: "paged_tensor",
				Shape: []any{
					2,
					4,
					8,
				},
				Config: map[string]any{
					"page_size":  4,
					"page_count": 2,
				},
			},
		})
		convey.So(err, convey.ShouldBeNil)

		executor := NewExecutor(ExecutorOptions{
			Backend: backend,
			State:   state,
		})
		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op:    "graph.call",
					Graph: "model",
					Out: map[string]string{
						"cache": "state.cache",
					},
				},
			},
		}

		graphs := map[string]*ast.Graph{
			"model": {},
		}

		err = executor.Run(context.Background(), program, graphs, map[string]any{"model": "compute"})
		convey.So(err, convey.ShouldBeNil)

		value, ok := state.Get("cache")
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(value, convey.ShouldEqual, cache)
	})
}

func TestExecutorRunLoopCountStopsOnSampleStopToken(testingObject *testing.T) {
	convey.Convey("Given a loop_count decode body with stop token sampling", testingObject, func() {
		backend := &countingGraphBackend{}
		executor := NewExecutor(ExecutorOptions{
			Backend: backend,
			InitialValues: map[string]any{
				"input_ids": []int{1},
			},
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "control.loop_count",
					Loop: &ast.Loop{
						Repeat: "3",
					},
					Body: []ast.Step{
						{
							Op:    "graph.call",
							Graph: "model",
							In: map[string]string{
								"input_ids": "input_ids",
							},
							Out: map[string]string{
								"logits": "logits",
							},
						},
						{
							Op: "sampling.topk_sample",
							In: map[string]string{
								"value": "logits",
							},
							Config: map[string]any{
								"top_k":          1,
								"stop_token_ids": []any{float64(2)},
							},
							Out: map[string]string{
								"value": "next_token",
							},
						},
						{
							Op: "value.append",
							In: map[string]string{
								"value": "next_token",
							},
							Config: map[string]any{
								"target": "input_ids",
							},
						},
					},
				},
			},
		}

		graphs := map[string]*ast.Graph{
			"model": {
				Outputs: map[string]string{"logits": "logits"},
			},
		}

		err := executor.Run(context.Background(), program, graphs, map[string]any{"model": "compute"})

		convey.So(err, convey.ShouldBeNil)
		convey.So(backend.calls, convey.ShouldEqual, 1)
	})
}

func TestExecutorRunTopKSampleAcceptsTensorLogits(testingObject *testing.T) {
	convey.Convey("Given top-k sampling with tensor logits", testingObject, func() {
		shape, err := tensor.NewShape([]int{3})
		convey.So(err, convey.ShouldBeNil)

		logits, err := tensor.New(shape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		logitValues, err := logits.Float32Native()
		convey.So(err, convey.ShouldBeNil)

		logitValues[0] = 0
		logitValues[1] = 1
		logitValues[2] = 2

		executor := NewExecutor(ExecutorOptions{
			InitialValues: map[string]any{
				"logits": logits,
			},
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "sampling.topk_sample",
					In: map[string]string{
						"value": "logits",
					},
					Config: map[string]any{
						"top_k": 1,
					},
					Out: map[string]string{
						"value": "next_token",
					},
				},
			},
		}

		convey.Convey("It should sample without converting through nil", func() {
			err := executor.Run(context.Background(), program, nil, nil)

			convey.So(err, convey.ShouldBeNil)
		})
	})
}

func TestExecutorRunLoopUntilEOFStopsOnEmptyLine(testingObject *testing.T) {
	convey.Convey("Given a loop_until_eof body that reads an empty line", testingObject, func() {
		host := &encodeCaptureHost{line: ""}
		executor := NewExecutor(ExecutorOptions{
			Host: host,
		})

		program := &ast.Program{
			Steps: []ast.Step{
				{
					Op: "control.loop_until_eof",
					Body: []ast.Step{
						{
							Op: "io.read_line",
							Out: map[string]string{
								"value": "user_text",
							},
						},
						{
							Op: "tokenizer.encode",
							In: map[string]string{
								"text": "user_text",
							},
							Out: map[string]string{
								"value": "input_ids",
							},
						},
					},
				},
			},
		}

		err := executor.Run(context.Background(), program, nil, nil)

		convey.So(err, convey.ShouldBeNil)
		convey.So(host.encodeCalls, convey.ShouldEqual, 0)
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

func TestMaterializeZeroInitializedStateTensor(testingObject *testing.T) {
	convey.Convey("Given a tensor state declaration without an initializer", testingObject, func() {
		declarations := []ast.StateDeclaration{
			{
				Name:  "position_offset",
				Type:  "tensor",
				Shape: []any{1},
			},
		}
		state, err := NewStateStore(declarations)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should materialize a zero-filled tensor matching the declared shape", func() {
			err := MaterializeStateTensors(state, declarations, tensor.NewHostBackend(), dtype.Float32)
			convey.So(err, convey.ShouldBeNil)

			value, ok := state.Get("position_offset")
			convey.So(ok, convey.ShouldBeTrue)

			stateTensor, ok := value.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(stateTensor.Shape().Dims(), convey.ShouldResemble, []int{1})

			values, err := stateTensor.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float32{0})
		})
	})
}

func TestMaterializeStatePagedTensors(testingObject *testing.T) {
	convey.Convey("Given generic paged tensor state and a runtime dtype", testingObject, func() {
		declarations := []ast.StateDeclaration{
			{
				Name: "pages",
				Type: "paged_tensor",
				Shape: []any{
					2,
					4,
				},
				Config: map[string]any{
					"page_size":  4,
					"page_count": 2,
				},
			},
		}
		state, err := NewStateStore(declarations)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should attach resident tensor storage to the paged handle", func() {
			err := MaterializeStateTensors(state, declarations, tensor.NewHostBackend(), dtype.BFloat16)
			convey.So(err, convey.ShouldBeNil)

			value, ok := state.Get("pages")
			convey.So(ok, convey.ShouldBeTrue)

			pages, ok := value.(*PagedTensorState)
			convey.So(ok, convey.ShouldBeTrue)

			stateTensor, ok := pages.Storage.(tensor.Tensor)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(stateTensor.DType(), convey.ShouldEqual, dtype.BFloat16)
			convey.So(stateTensor.Shape().Dims(), convey.ShouldResemble, []int{2, 4})
		})
	})
}

func TestNewStateStorePagedState(testingObject *testing.T) {
	convey.Convey("Given generic paged state declarations", testingObject, func() {
		state, err := NewStateStore([]ast.StateDeclaration{
			{
				Name: "pages",
				Type: "paged_tensor",
				Shape: []any{
					2,
					4,
					8,
				},
				Config: map[string]any{
					"page_size":  4,
					"page_count": 2,
				},
			},
			{
				Name: "table",
				Type: "page_table",
				Config: map[string]any{
					"capacity": 8,
				},
			},
		})

		convey.So(err, convey.ShouldBeNil)

		pages, ok := state.slots["pages"].(*PagedTensorState)
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(pages.Shape, convey.ShouldResemble, []int{2, 4, 8})
		convey.So(pages.PageSize, convey.ShouldEqual, 4)
		convey.So(pages.PageCount, convey.ShouldEqual, 2)

		table, ok := state.slots["table"].(*PageTableState)
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(table.Capacity, convey.ShouldEqual, 8)
		convey.So(table.Pages, convey.ShouldResemble, []int32{})
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

type countingGraphBackend struct {
	calls int
}

func (backend *countingGraphBackend) CallGraph(
	ctx context.Context,
	request GraphCallRequest,
) (GraphCallResult, error) {
	_ = ctx
	_ = request
	backend.calls++

	return GraphCallResult{Outputs: map[string]any{"logits": []float32{0, 1, 2}}}, nil
}

type graphOutputBackend struct {
	outputs map[string]any
}

func (backend *graphOutputBackend) CallGraph(
	ctx context.Context,
	request GraphCallRequest,
) (GraphCallResult, error) {
	_ = ctx
	_ = request

	return GraphCallResult{Outputs: backend.outputs}, nil
}

type encodeCaptureHost struct {
	request      EncodeRequest
	imageRequest WriteImageRequest
	tokens       []int
	line         string
	encodeCalls  int
}

func (host *encodeCaptureHost) ReadLine(ctx context.Context) (string, error) {
	_ = ctx

	return host.line, nil
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
	host.encodeCalls++

	if host.tokens != nil {
		return host.tokens, nil
	}

	return []int{1, 2, 3}, nil
}
