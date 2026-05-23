package compiler

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestComponentVariablesFromHubConfig(t *testing.T) {
	convey.Convey("Given a Qwen3 text encoder hub config", t, func() {
		hubConfig := map[string]any{
			"hidden_size":          float64(2560),
			"intermediate_size":    float64(9728),
			"num_hidden_layers":    float64(36),
			"num_attention_heads":  float64(32),
			"num_key_value_heads":  float64(8),
			"head_dim":             float64(128),
			"vocab_size":           float64(151936),
			"rope_theta":           float64(1000000),
			"rms_norm_eps":         1e-06,
		}

		convey.Convey("It should derive projection and prompt layer variables", func() {
			variables := componentVariablesFromHubConfig(hubConfig)

			convey.So(variables["hidden_size"], convey.ShouldEqual, 2560)
			convey.So(variables["q_proj_out"], convey.ShouldEqual, 4096)
			convey.So(variables["kv_proj_out"], convey.ShouldEqual, 1024)
			convey.So(variables["prompt_layer_a"], convey.ShouldEqual, 9)
			convey.So(variables["prompt_layer_b"], convey.ShouldEqual, 18)
			convey.So(variables["prompt_layer_c"], convey.ShouldEqual, 27)
			convey.So(variables["eps"], convey.ShouldEqual, 1e-06)
		})
	})
}
