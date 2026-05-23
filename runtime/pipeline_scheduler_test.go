package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestSchedulersFromPipelineInclude(t *testing.T) {
	convey.Convey("Given a program with a Hub pipeline include", t, func() {
		program := &ast.Program{
			Variables: map[string]any{
				"num_inference_steps": 4,
			},
			IncludeObjects: map[string]any{
				"flux2klein": map[string]any{
					"kind": "Pipeline",
					"system": map[string]any{
						"components": map[string]any{
							"scheduler": map[string]any{
								"class_name": "FlowMatchEulerDiscreteScheduler",
								"config": map[string]any{
									"num_train_timesteps":  1000,
									"shift":                3.0,
									"use_dynamic_shifting": true,
									"time_shift_type":      "exponential",
								},
							},
						},
					},
				},
			},
		}

		convey.Convey("It should construct schedulers from the pipeline component", func() {
			schedulers, err := SchedulersFromProgram(program)

			convey.So(err, convey.ShouldBeNil)
			convey.So(schedulers, convey.ShouldContainKey, "scheduler")

			scheduler := schedulers["scheduler"]

			convey.So(scheduler.Steps, convey.ShouldEqual, 4)
			convey.So(scheduler.NumTrainTimesteps, convey.ShouldEqual, 1000)
			convey.So(scheduler.Shift, convey.ShouldEqual, 3.0)
			convey.So(scheduler.UseDynamicShift, convey.ShouldBeTrue)
			convey.So(scheduler.TimeShiftType, convey.ShouldEqual, "exponential")
		})
	})

	convey.Convey("Given explicit program schedulers", t, func() {
		program := &ast.Program{
			Schedulers: map[string]ast.SchedulerDeclaration{
				"custom": {
					Type: "flow_match_euler_discrete",
					Config: map[string]any{
						"steps":                8,
						"num_train_timesteps":  1000,
						"shift":                3.0,
						"use_dynamic_shifting": true,
						"time_shift_type":      "exponential",
					},
				},
			},
			IncludeObjects: map[string]any{
				"flux2klein": map[string]any{
					"kind": "Pipeline",
					"system": map[string]any{
						"components": map[string]any{
							"scheduler": map[string]any{
								"class_name": "FlowMatchEulerDiscreteScheduler",
								"config":     map[string]any{},
							},
						},
					},
				},
			},
		}

		convey.Convey("It should prefer yaml schedulers over pipeline includes", func() {
			schedulers, err := SchedulersFromProgram(program)

			convey.So(err, convey.ShouldBeNil)
			convey.So(schedulers, convey.ShouldContainKey, "custom")
			convey.So(schedulers, convey.ShouldNotContainKey, "scheduler")
			convey.So(schedulers["custom"].Steps, convey.ShouldEqual, 8)
		})
	})
}

func TestSchedulerNameFromConfig(t *testing.T) {
	convey.Convey("Given scheduler step config", t, func() {
		convey.Convey("It should default to scheduler", func() {
			convey.So(schedulerNameFromConfig(nil), convey.ShouldEqual, "scheduler")
			convey.So(schedulerNameFromConfig(map[string]any{}), convey.ShouldEqual, "scheduler")
		})

		convey.Convey("It should honor explicit scheduler names", func() {
			convey.So(schedulerNameFromConfig(map[string]any{
				"scheduler": "custom",
			}), convey.ShouldEqual, "custom")
		})
	})
}
