package expand

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/manifesto/ast"
)

func TestRecipe_Topology(t *testing.T) {
	convey.Convey("Given a recipe topology with one repeated layer", t, func() {
		expander := NewRecipe(nil)
		recipe := &ast.Recipe{
			Name: "fixture",
			Config: map[string]ast.Binding{
				"num_layers": {Config: "num_hidden_layers"},
			},
			Topology: &ast.Topology{
				Inputs: []string{"input_ids"},
				Nodes: []ast.Node{
					{
						Repeat: "${num_layers}",
						Index:  "layer",
						Template: []ast.Node{
							{
								ID:  "layers.${layer}.norm",
								Op:  "math.rmsnorm",
								In:  []string{"hidden_${layer}"},
								Out: []string{"hidden_${next_layer}"},
							},
						},
					},
				},
			},
		}

		convey.Convey("It should unroll repeat blocks using config bindings", func() {
			topology, err := expander.Topology(recipe, map[string]any{
				"num_hidden_layers": 2,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(len(topology.Nodes), convey.ShouldEqual, 2)
			convey.So(topology.Nodes[0].ID, convey.ShouldEqual, "layers.0.norm")
			convey.So(topology.Nodes[1].ID, convey.ShouldEqual, "layers.1.norm")
		})
	})
}
