package runtime

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
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
