package compiler

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestSchedulerTypeFromHubClass(t *testing.T) {
	convey.Convey("Given a FlowMatchEulerDiscreteScheduler hub config", t, func() {
		hubConfig := map[string]any{
			"_class_name": "FlowMatchEulerDiscreteScheduler",
		}

		convey.Convey("It should map to the runtime scheduler type", func() {
			convey.So(schedulerTypeFromHubClass(hubConfig), convey.ShouldEqual, "flow_match_euler_discrete")
		})
	})
}

func TestPreserveSchedulerRuntimeFields(t *testing.T) {
	convey.Convey("Given runtime scheduler overrides", t, func() {
		scheduler := map[string]any{
			"source":              "repo/model",
			"path":                "scheduler",
			"num_inference_steps": 4,
			"guidance_scale":      1.0,
			"shift":               99.0,
		}

		convey.Convey("It should preserve inference knobs across hub merge", func() {
			preserved := preserveSchedulerRuntimeFields(scheduler)

			convey.So(preserved["num_inference_steps"], convey.ShouldEqual, 4)
			convey.So(preserved["guidance_scale"], convey.ShouldEqual, 1.0)
			convey.So(preserved, convey.ShouldNotContainKey, "shift")
		})
	})
}

func TestLatentDownsampleFromVAEConfig(t *testing.T) {
	convey.Convey("Given a VAE config with four block_out_channels", t, func() {
		vaeConfig := map[string]any{
			"block_out_channels": []any{128, 256, 512, 512},
		}

		convey.Convey("It should derive latent_downsample as vae_scale_factor * 2", func() {
			downsample, err := latentDownsampleFromVAEConfig(vaeConfig)

			convey.So(err, convey.ShouldBeNil)
			convey.So(downsample, convey.ShouldEqual, 16)
		})
	})
}
