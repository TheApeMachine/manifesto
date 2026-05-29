package runtime

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/optimizer"
)

type compileTestResolver struct {
	payload []byte
}

func (resolver *compileTestResolver) ResolveInclude(
	ctx context.Context,
	include compiler.IncludeSource,
) ([]byte, error) {
	_ = ctx
	_ = include

	return resolver.payload, nil
}

func TestExecutionPlanFromCompiledFusionGraph(test *testing.T) {
	convey.Convey("Given CompileAssets output with fusion enabled", test, func() {
		programYAML := []byte(`kind: program
name: fusion
include:
  model: hf://example/fusion-model
main:
  - op: graph.call
    graph: model
`)

		blockYAML := []byte(`kind: Block
category: model
name: fusion-model
system:
  topology:
    inputs:
      - x
      - y
    outputs:
      result: relu_out
    nodes:
      - id: add_out
        op: math.add
        in: [x, y]
        out: [add_out]
      - id: sigmoid_out
        op: activation.sigmoid
        in: [add_out]
        out: [sigmoid_out]
      - id: relu_out
        op: activation.relu
        in: [sigmoid_out]
        out: [relu_out]
`)

		programCompiler, err := compiler.NewProgramCompiler(context.Background(), compiler.NewPool(nil))
		convey.So(err, convey.ShouldBeNil)

		programCompiler = programCompiler.
			WithIncludeResolver(&compileTestResolver{payload: blockYAML}).
			WithPlannerBindings(ir.SymbolMap{"N": 4})

		output, err := programCompiler.CompileAssets(context.Background(), compiler.CompileInput{
			ProgramYAML: programYAML,
		}, nil)

		convey.So(err, convey.ShouldBeNil)

		plan, err := NewExecutionPlan("model", output.ComputeGraphs["model"])

		convey.Convey("ExecutionPlan references the fused ast node id", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(output.Graphs["model"].Nodes), convey.ShouldEqual, 1)
			convey.So(output.Graphs["model"].Nodes[0].Op, convey.ShouldEqual, optimizer.FuseOp)
			convey.So(len(plan.Layers), convey.ShouldEqual, 1)
			convey.So(plan.Layers[0][0], convey.ShouldEqual, output.Graphs["model"].Nodes[0].ID)
		})
	})
}
