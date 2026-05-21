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

	convey.Convey("Given a program manifest with a flattened include object", t, func() {
		raw := []byte(`
kind: program
name: diffusion
category: runtime
include:
  flux2klein:
    kind: Block
    category: model
    name: FLUX.2 Klein 4B
    inputs:
      - name: hidden_states
        type: tensor
    system:
      runtime:
        backend: metal
main:
  - op: graph.call
    graph: flux2klein
    out:
      sample: sample
`)
		parser := NewParser()

		convey.Convey("It should preserve the include object separately from string includes", func() {
			program, err := parser.Program(raw)

			convey.So(err, convey.ShouldBeNil)
			convey.So(program.Includes, convey.ShouldBeEmpty)
			convey.So(program.IncludeObjects, convey.ShouldContainKey, "flux2klein")
			convey.So(program.Steps[0].Graph, convey.ShouldEqual, "flux2klein")
		})
	})
}

func TestParser_Manifest(t *testing.T) {
	convey.Convey("Given an arbitrary new-shape manifest", t, func() {
		raw := []byte(`
kind: Block
name: FLUX.2 Klein 4B
category: model
include:
  topology: model.architecture.flux2
system:
  runtime:
    backend: metal
`)
		parser := NewParser()

		convey.Convey("It should preserve kind, category, include, and raw fields", func() {
			document, err := parser.Manifest(raw)

			convey.So(err, convey.ShouldBeNil)
			convey.So(document.Kind, convey.ShouldEqual, "Block")
			convey.So(document.Name, convey.ShouldEqual, "FLUX.2 Klein 4B")
			convey.So(document.Category, convey.ShouldEqual, "model")
			convey.So(document.Include["topology"], convey.ShouldEqual, "model.architecture.flux2")
			convey.So(document.Raw["system"], convey.ShouldNotBeNil)
		})
	})
}
