package asset

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/expand"
	"github.com/theapemachine/manifesto/lower"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/registry"
	"gopkg.in/yaml.v3"
)

func TestFlux2Transformer2DModel(t *testing.T) {
	convey.Convey("Given the FLUX.2 transformer config fields from Hugging Face", t, func() {
		catalogInstance := catalog.NewFS(TemplateFS())
		registryInstance, err := registry.NewRegistry(catalogInstance)
		convey.So(err, convey.ShouldBeNil)

		recipe, err := registryInstance.Recipe("Flux2Transformer2DModel")
		convey.So(err, convey.ShouldBeNil)

		config := map[string]any{
			"attention_head_dim":  128,
			"eps":                 1e-6,
			"in_channels":         128,
			"joint_attention_dim": 7680,
			"mlp_ratio":           3.0,
			"num_attention_heads": 24,
			"num_layers":          5,
			"num_single_layers":   20,
			"rope_theta":          2000,
		}

		convey.Convey("It should derive the topology without requiring hidden_size", func() {
			topology, err := expand.NewRecipe(catalogInstance).Topology(recipe, config)

			convey.So(err, convey.ShouldBeNil)
			convey.So(topology.Nodes, convey.ShouldNotBeEmpty)
		})

		convey.Convey("It should preserve fused safetensors slices", func() {
			topology, err := expand.NewRecipe(catalogInstance).Topology(recipe, config)
			convey.So(err, convey.ShouldBeNil)

			graph, err := lower.NewLowerer().Topology(topology, dtype.Float32)
			convey.So(err, convey.ShouldBeNil)

			found := false

			for _, node := range graph.Nodes {
				if node.ID != "single_transformer_blocks.0.attn.to_q" {
					continue
				}

				found = true
				convey.So(node.Weights, convey.ShouldNotBeNil)
				convey.So(node.Weights.TensorName, convey.ShouldEqual, "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight")
				convey.So(node.Weights.Slice, convey.ShouldNotBeNil)
				convey.So(node.Weights.Slice.Axis, convey.ShouldEqual, "output")
				convey.So(node.Weights.Slice.Start, convey.ShouldEqual, 0)
			}

			convey.So(found, convey.ShouldBeTrue)
		})

		convey.Convey("It should wire timestep conditioning into the final adaptive norm", func() {
			topology, err := expand.NewRecipe(catalogInstance).Topology(recipe, config)
			convey.So(err, convey.ShouldBeNil)

			graph, err := lower.NewLowerer().Topology(topology, dtype.Float32)
			convey.So(err, convey.ShouldBeNil)

			required := map[string]bool{
				"time_guidance_embed.time_proj":                  false,
				"time_guidance_embed.timestep_embedder.linear_1": false,
				"time_guidance_embed.timestep_embedder.linear_2": false,
				"single_stream_modulation.linear":                false,
				"single_transformer_blocks.0.attn.norm_q":        false,
				"single_transformer_blocks.0.attn.norm_k":        false,
				"norm_out.linear":                                false,
				"norm_out":                                       false,
			}

			for _, node := range graph.Nodes {
				if _, ok := required[node.ID]; !ok {
					continue
				}

				required[node.ID] = true
			}

			for _, found := range required {
				convey.So(found, convey.ShouldBeTrue)
			}
		})
	})
}

func TestDiffusionRuntimeIncludesFlux2Klein(t *testing.T) {
	convey.Convey("Given the diffusion runtime manifest", t, func() {
		raw, err := ReadFile("runtime/diffusion.yml")
		convey.So(err, convey.ShouldBeNil)

		document := make(map[string]any)
		err = yaml.Unmarshal(raw, &document)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should include the model manifest instead of component graph manifests", func() {
			include := document["include"].(map[string]any)

			convey.So(include, convey.ShouldContainKey, "flux2klein")
			convey.So(include["flux2klein"], convey.ShouldEqual, "model.diffusion.flux-2-klein-4b")
			convey.So(include, convey.ShouldNotContainKey, "text_encoder")
			convey.So(include, convey.ShouldNotContainKey, "transformer")
			convey.So(include, convey.ShouldNotContainKey, "vae")
		})

		convey.Convey("It should express scheduler updates as program operations", func() {
			mainSteps := document["main"].([]any)
			operations := make(map[string]bool)
			removedSchedulerOperation := "scheduler." + "step"

			collectOperations(mainSteps, operations)

			convey.So(operations, convey.ShouldContainKey, "scheduler.delta")
			convey.So(operations, convey.ShouldContainKey, "math.axpy")
			convey.So(operations, convey.ShouldNotContainKey, removedSchedulerOperation)
		})
	})
}

func collectOperations(steps []any, operations map[string]bool) {
	for _, rawStep := range steps {
		step, ok := rawStep.(map[string]any)

		if !ok {
			continue
		}

		if operation, ok := step["op"].(string); ok {
			operations[operation] = true
		}

		childSteps, ok := step["steps"].([]any)

		if !ok {
			continue
		}

		collectOperations(childSteps, operations)
	}
}

func TestAutoencoderKLFlux2(t *testing.T) {
	convey.Convey("Given the FLUX.2 VAE architecture class", t, func() {
		catalogInstance := catalog.NewFS(TemplateFS())
		registryInstance, err := registry.NewRegistry(catalogInstance)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should resolve a topology for HF VAE components", func() {
			recipe, err := registryInstance.Recipe("AutoencoderKLFlux2")

			convey.So(err, convey.ShouldBeNil)
			convey.So(recipe.Topology, convey.ShouldNotBeNil)
			convey.So(recipe.Topology.Nodes, convey.ShouldNotBeEmpty)
		})
	})
}

func TestFlux2TextEncoderSwiGLUShape(t *testing.T) {
	convey.Convey("Given the FLUX.2 text encoder topology", t, func() {
		raw, err := ReadFile("model/diffusion/flux-2-klein-4b-text-encoder.yml")
		convey.So(err, convey.ShouldBeNil)

		block, err := parse.BlockModelFromYAML(raw)
		convey.So(err, convey.ShouldBeNil)

		topology, err := block.TopologyAST()
		convey.So(err, convey.ShouldBeNil)

		topology, err = expand.NewRecipe(catalog.NewFS(TemplateFS())).ExpandTopology(topology)
		convey.So(err, convey.ShouldBeNil)

		graph, err := lower.NewLowerer().Topology(topology, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should halve packed gate/up activations", func() {
			found := false

			for _, node := range graph.Nodes {
				if node.ID != "swiglu_0" {
					continue
				}

				found = true
				convey.So(node.ValueType.Shape, convey.ShouldResemble, []int64{-1, -1, 9728})
			}

			convey.So(found, convey.ShouldBeTrue)
		})
	})
}

func TestFlux2TextEncoderConcatArity(t *testing.T) {
	convey.Convey("Given the FLUX.2 text encoder topology", t, func() {
		raw, err := ReadFile("model/diffusion/flux-2-klein-4b-text-encoder.yml")
		convey.So(err, convey.ShouldBeNil)

		block, err := parse.BlockModelFromYAML(raw)
		convey.So(err, convey.ShouldBeNil)

		topology, err := block.TopologyAST()
		convey.So(err, convey.ShouldBeNil)

		topology, err = expand.NewRecipe(catalog.NewFS(TemplateFS())).ExpandTopology(topology)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should only use binary concat nodes supported by Metal", func() {
			for _, node := range topology.Nodes {
				if node.Op != "shape.concat" {
					continue
				}

				convey.So(len(node.In), convey.ShouldEqual, 2)
			}
		})
	})
}
