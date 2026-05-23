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

	convey.Convey("Given a recipe topology with numeric config interpolation", t, func() {
		expander := NewRecipe(nil)
		recipe := &ast.Recipe{
			Name: "fixture",
			Config: map[string]ast.Binding{
				"hidden_size": {Config: "hidden_size"},
			},
			Topology: &ast.Topology{
				Inputs: []string{"input"},
				Nodes: []ast.Node{
					{
						ID:  "linear",
						Op:  "projection.linear",
						In:  []string{"input"},
						Out: []string{"output"},
						Config: map[string]any{
							"out_features": "${include.hidden_size}",
						},
					},
				},
			},
		}

		convey.Convey("It should preserve the numeric binding type", func() {
			topology, err := expander.Topology(recipe, map[string]any{
				"hidden_size": 3072,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Nodes[0].Config["out_features"], convey.ShouldEqual, 3072)
		})
	})

	convey.Convey("Given a repeated topology with an offset handoff", t, func() {
		expander := NewRecipe(nil)
		recipe := &ast.Recipe{
			Name: "fixture",
			Config: map[string]ast.Binding{
				"num_layers": {Config: "num_layers"},
			},
			Topology: &ast.Topology{
				Inputs: []string{"h_5"},
				Nodes: []ast.Node{
					{
						Repeat: 2,
						Index:  "layer",
						Offset: "${include.num_layers}",
						Template: []ast.Node{
							{
								ID:  "block.${layer}",
								Op:  "math.add",
								In:  []string{"h_${offset_layer}", "delta_${layer}"},
								Out: []string{"h_${next_offset_layer}"},
							},
						},
					},
				},
			},
		}

		convey.Convey("It should wire each block to the next offset output", func() {
			topology, err := expander.Topology(recipe, map[string]any{
				"num_layers": 5,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Nodes[0].In[0], convey.ShouldEqual, "h_5")
			convey.So(topology.Nodes[0].Out[0], convey.ShouldEqual, "h_6")
			convey.So(topology.Nodes[1].In[0], convey.ShouldEqual, "h_6")
			convey.So(topology.Nodes[1].Out[0], convey.ShouldEqual, "h_7")
		})
	})

	convey.Convey("Given a topology node with shape arrays in config", t, func() {
		expander := NewRecipe(nil)
		topology := &ast.Topology{
			Inputs: []string{"latents"},
			Nodes: []ast.Node{
				{
					ID:  "vae.unpack.grid",
					Op:  "shape.reshape",
					In:  []string{"latents"},
					Out: []string{"packed_grid"},
					Config: map[string]any{
						"shape": []any{1, "${packed_side}", "${packed_side}", 128},
					},
				},
			},
		}

		convey.Convey("It should interpolate every shape dimension", func() {
			expanded, err := expander.ExpandTopologyWithVariables(topology, map[string]any{
				"packed_side": 16,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(expanded.Nodes[0].Config["shape"], convey.ShouldResemble, []any{1, 16, 16, 128})
		})
	})
}
