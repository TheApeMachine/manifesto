package compiler

import (
	"context"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestLowerTopologyFlatChain(t *testing.T) {
	convey.Convey("Given a flat topology of three nodes", t, func() {
		topology := &ast.Topology{
			Inputs: []string{"input_ids"},
			Nodes: []ast.Node{
				{
					ID:  "embed",
					Op:  "embedding.token",
					In:  []string{"input_ids"},
					Out: []string{"hidden"},
					Config: map[string]any{
						"vocab_size": 128256,
						"d_model":    2048,
					},
				},
				{
					ID:  "norm",
					Op:  "math.rmsnorm",
					In:  []string{"hidden"},
					Out: []string{"normed"},
					Weights: &ast.WeightSpec{
						Weight: "model.norm.weight",
					},
				},
				{
					ID:  "out",
					Op:  "projection.linear",
					In:  []string{"normed"},
					Out: []string{"logits"},
				},
			},
		}

		lowered, err := LowerTopology(topology)

		convey.So(err, convey.ShouldBeNil)
		convey.So(lowered, convey.ShouldNotBeNil)

		convey.Convey("Then the ast.Graph carries op kinds, inputs, attrs, and weights", func() {
			convey.So(lowered.AST.Inputs, convey.ShouldResemble, []string{"input_ids"})
			convey.So(len(lowered.AST.Nodes), convey.ShouldEqual, 3)
			convey.So(lowered.AST.Nodes[0].ID, convey.ShouldEqual, "embed")
			convey.So(lowered.AST.Nodes[0].Op, convey.ShouldEqual, "embedding.token")
			convey.So(lowered.AST.Nodes[0].Inputs, convey.ShouldResemble, []string{"input_ids"})
			convey.So(lowered.AST.Nodes[0].Attributes["vocab_size"], convey.ShouldEqual, 128256)
			convey.So(lowered.AST.Nodes[1].Weights, convey.ShouldNotBeNil)
			convey.So(lowered.AST.Nodes[1].Weights.TensorName, convey.ShouldEqual, "model.norm.weight")
		})

		convey.Convey("And the dag.Graph forms a verifiable topology with layered execution", func() {
			convey.So(lowered.DAG.Verify(), convey.ShouldBeNil)

			layers, err := lowered.DAG.TopologyLayers()

			convey.So(err, convey.ShouldBeNil)
			convey.So(len(layers), convey.ShouldEqual, 4)
			convey.So(layers[0][0].ID(), convey.ShouldEqual, "input_ids")
			convey.So(layers[1][0].ID(), convey.ShouldEqual, "embed")
			convey.So(layers[2][0].ID(), convey.ShouldEqual, "norm")
			convey.So(layers[3][0].ID(), convey.ShouldEqual, "out")
		})
	})
}

func TestLowerTopologyExpandsRepeat(t *testing.T) {
	convey.Convey("Given a topology with a control.repeat block", t, func() {
		topology := &ast.Topology{
			Inputs: []string{"h_0"},
			Nodes: []ast.Node{
				{
					ID:     "transformer_layers",
					Op:     "control.repeat",
					In:     []string{"h_0"},
					Out:    []string{"h_2"},
					Repeat: 2,
					Index:  "i",
					Template: []ast.Node{
						{
							ID:  "norm_${i}",
							Op:  "math.rmsnorm",
							In:  []string{"h_${i}"},
							Out: []string{"normed_${i}"},
							Weights: &ast.WeightSpec{
								Weight: "model.layers.${i}.norm.weight",
							},
						},
						{
							ID:  "linear_${i}",
							Op:  "projection.linear",
							In:  []string{"normed_${i}"},
							Out: []string{"h_${i+1}"},
							Weights: &ast.WeightSpec{
								Weight: "model.layers.${i}.linear.weight",
							},
						},
					},
				},
			},
		}

		lowered, err := LowerTopology(topology)

		convey.So(err, convey.ShouldBeNil)
		convey.So(lowered, convey.ShouldNotBeNil)

		convey.Convey("Then the repeat block materializes one ast.Node per (iteration, template)", func() {
			convey.So(len(lowered.AST.Nodes), convey.ShouldEqual, 4)
			convey.So(lowered.AST.Nodes[0].ID, convey.ShouldEqual, "norm_0")
			// Inputs are resolved to producer node IDs: "h_0" is a graph
			// input so its producer ID is the input name itself.
			convey.So(lowered.AST.Nodes[0].Inputs, convey.ShouldResemble, []string{"h_0"})
			convey.So(lowered.AST.Nodes[0].Weights.TensorName, convey.ShouldEqual, "model.layers.0.norm.weight")
			convey.So(lowered.AST.Nodes[1].ID, convey.ShouldEqual, "linear_0")
			convey.So(lowered.AST.Nodes[3].ID, convey.ShouldEqual, "linear_1")
			// "normed_1" was produced by node "norm_1" via its `out`
			// declaration — after resolution this becomes "norm_1".
			convey.So(lowered.AST.Nodes[3].Inputs, convey.ShouldResemble, []string{"norm_1"})
			convey.So(lowered.AST.Nodes[3].Weights.TensorName, convey.ShouldEqual, "model.layers.1.linear.weight")
		})

		convey.Convey("And the dag.Graph wires norm_1 to the linear_0 output h_1", func() {
			convey.So(lowered.DAG.Verify(), convey.ShouldBeNil)
		})
	})
}

func TestLowerTopologyRejectsUnknownInput(t *testing.T) {
	convey.Convey("Given a topology that references an unknown input", t, func() {
		topology := &ast.Topology{
			Inputs: []string{"input_ids"},
			Nodes: []ast.Node{
				{
					ID: "norm",
					Op: "math.rmsnorm",
					In: []string{"missing"},
				},
			},
		}

		_, err := LowerTopology(topology)

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "unknown input")
	})
}

func TestParseHFReference(t *testing.T) {
	convey.Convey("Given hf:// URIs", t, func() {
		repo, component, ok := ParseHFReference("hf://meta-llama/Llama-3.2-1B-Instruct")
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(repo, convey.ShouldEqual, "meta-llama/Llama-3.2-1B-Instruct")
		convey.So(component, convey.ShouldEqual, "")

		repo, component, ok = ParseHFReference("hf://black-forest-labs/FLUX.2-klein-4B#transformer")
		convey.So(ok, convey.ShouldBeTrue)
		convey.So(repo, convey.ShouldEqual, "black-forest-labs/FLUX.2-klein-4B")
		convey.So(component, convey.ShouldEqual, "transformer")

		_, _, ok = ParseHFReference("model/diffusion/flux.yml")
		convey.So(ok, convey.ShouldBeFalse)
	})
}

func TestCompileAssetsWithoutResolverFails(t *testing.T) {
	convey.Convey("Given a program YAML that declares an hf:// include and no resolver", t, func() {
		programYAML := []byte(`kind: program
name: chat
include:
  model: hf://meta-llama/Llama-3.2-1B-Instruct
main:
  - in: <stdin>
    op: io.read_line
    out: user_text
`)

		programCompiler, err := NewProgramCompiler(NewPool(nil))

		convey.So(err, convey.ShouldBeNil)

		_, err = programCompiler.CompileAssets(context.Background(), CompileInput{
			ProgramYAML: programYAML,
		}, nil)

		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "no IncludeResolver is configured")
	})
}

func TestCompileAssetsResolvesInclude(t *testing.T) {
	convey.Convey("Given a program with an include and a fake resolver", t, func() {
		programYAML := []byte(`kind: program
name: chat
include:
  model: hf://example/fake-model
main:
  - in: <stdin>
    op: io.read_line
    out: user_text
`)

		blockYAML := []byte(`kind: Block
category: model
op: block.model.fake
name: fake-model
outputs:
  - name: logits
system:
  topology:
    inputs:
      - input_ids
    nodes:
      - id: embed
        op: embedding.token
        in:
          - input_ids
        out:
          - hidden
      - id: norm
        op: math.rmsnorm
        in:
          - hidden
        out:
          - logits
`)

		resolver := &fakeResolver{payload: blockYAML}
		programCompiler, _ := NewProgramCompiler(NewPool(nil))
		programCompiler = programCompiler.
			WithIncludeResolver(resolver).
			DisableTyper().
			DisableOptimizer().
			DisableCodegen()

		output, err := programCompiler.CompileAssets(context.Background(), CompileInput{
			ProgramYAML: programYAML,
		}, nil)

		convey.So(err, convey.ShouldBeNil)
		convey.So(output.Graphs["model"], convey.ShouldNotBeNil)
		convey.So(output.ComputeGraphs["model"], convey.ShouldNotBeNil)
		convey.So(len(output.Graphs["model"].Nodes), convey.ShouldEqual, 2)
		convey.So(output.Graphs["model"].Nodes[0].ID, convey.ShouldEqual, "embed")
		convey.So(output.Graphs["model"].Outputs["logits"], convey.ShouldEqual, "norm")
		convey.So(resolver.calls, convey.ShouldEqual, 1)
		convey.So(resolver.lastInclude.Name, convey.ShouldEqual, "model")
		convey.So(resolver.lastInclude.Source, convey.ShouldEqual, "hf://example/fake-model")
	})
}

type fakeResolver struct {
	payload     []byte
	calls       int
	lastInclude IncludeSource
}

func (resolver *fakeResolver) ResolveInclude(ctx context.Context, include IncludeSource) ([]byte, error) {
	_ = ctx
	resolver.calls++
	resolver.lastInclude = include

	return resolver.payload, nil
}
