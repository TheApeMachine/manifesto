package parse

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestParser_Program(t *testing.T) {
	convey.Convey("Given a minimal program manifest", t, func() {
		raw := []byte(`
kind: Program
name: generate-image
includes:
  repo: black-forest-labs/FLUX.2-klein-4B
main:
  - in: stdin
    op: io.read_line
    out: prompt
`)
		parser := NewParser()

		convey.Convey("It should parse the program and repo include", func() {
			program, err := parser.Program(raw)
			convey.So(err, convey.ShouldBeNil)
			convey.So(program.Name, convey.ShouldEqual, "generate-image")
			convey.So(program.Includes["repo"], convey.ShouldEqual, "black-forest-labs/FLUX.2-klein-4B")
			convey.So(len(program.Steps), convey.ShouldEqual, 1)
			convey.So(program.Steps[0].Op, convey.ShouldEqual, "io.read_line")
		})
	})

	convey.Convey("Given a runtime program with top-level state and schedulers", t, func() {
		raw := []byte(`
kind: program
name: diffusion
category: runtime
include:
  transformer: hf://black-forest-labs/FLUX.2-klein-4B#transformer
state:
  - name: latents
    type: tensor
    shape: [1, 4096, 128]
    init: gaussian
schedulers:
  scheduler:
    type: flow_match_euler_discrete
    config:
      steps: 4
main:
  - op: scheduler.timesteps
    config:
      scheduler: scheduler
    out: timesteps
`)
		parser := NewParser()

		convey.Convey("It should parse the top-level runtime sections", func() {
			program, err := parser.Program(raw)

			convey.So(err, convey.ShouldBeNil)
			convey.So(program.Includes["transformer"], convey.ShouldEqual, "hf://black-forest-labs/FLUX.2-klein-4B#transformer")
			convey.So(len(program.State), convey.ShouldEqual, 1)
			convey.So(program.State[0].Name, convey.ShouldEqual, "latents")
			convey.So(program.Schedulers["scheduler"].Type, convey.ShouldEqual, "flow_match_euler_discrete")
			convey.So(program.Steps[0].Op, convey.ShouldEqual, "scheduler.timesteps")
		})
	})
}
