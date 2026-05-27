package parse

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestParserApplyVariables(testingObject *testing.T) {
	convey.Convey("Given a program manifest with variable references", testingObject, func() {
		raw := []byte(`
kind: program
name: diffusion
variables:
  steps: 4
  seed: 1337
  output_path: image.png
state:
  - name: latents
    type: tensor
    shape: [1, $variables.steps, 2]
    seed: $variables.seed
main:
  - repeat: $variables.steps
    steps:
      - op: random.normal
        config:
          shape: [1, $variables.steps, 2]
          seed: $variables.seed
        out: state.latents
  - op: io.write_image
    in:
      image: decoded
    config:
      path: $variables.output_path
`)
		parser := NewParser()

		convey.Convey("It should resolve variable references before execution", func() {
			program, err := parser.Program(raw)
			convey.So(err, convey.ShouldBeNil)
			convey.So(program.State[0].Shape, convey.ShouldResemble, []any{1, 4, 2})
			convey.So(program.State[0].Seed, convey.ShouldEqual, 1337)
			convey.So(program.Steps[0].Loop.Repeat, convey.ShouldEqual, "4")
			convey.So(program.Steps[0].Body[0].Config["shape"], convey.ShouldResemble, []any{1, 4, 2})
			convey.So(program.Steps[0].Body[0].Config["seed"], convey.ShouldEqual, 1337)
			convey.So(program.Steps[1].Config["path"], convey.ShouldEqual, "image.png")
		})
	})
}

func TestParserApplyVariablesRejectsMissing(testingObject *testing.T) {
	convey.Convey("Given a program manifest with an undeclared variable", testingObject, func() {
		raw := []byte(`
kind: program
name: diffusion
variables:
  steps: 4
main:
  - op: math.linspace
    config:
      count: $variables.missing
    out: sigmas
`)
		parser := NewParser()

		convey.Convey("It should fail during parsing", func() {
			_, err := parser.Program(raw)

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, `variable "missing"`)
		})
	})
}

func TestParserApplyVariablesDiffusionAsset(testingObject *testing.T) {
	convey.Convey("Given the diffusion runtime asset", testingObject, func() {
		path := filepath.Join("..", "asset", "template", "runtime", "diffusion.yml")
		raw, err := os.ReadFile(path)
		convey.So(err, convey.ShouldBeNil)

		parser := NewParser()

		convey.Convey("It should resolve scheduler setup variables", func() {
			program, err := parser.Program(raw)
			convey.So(err, convey.ShouldBeNil)
			convey.So(program.State[0].Type, convey.ShouldEqual, "counter")
			convey.So(program.Steps[3].Config["seed"], convey.ShouldEqual, 1337)
			convey.So(program.Steps[4].Config["count"], convey.ShouldEqual, 4)
			convey.So(program.Steps[5].Out["value"], convey.ShouldEqual, "state.timesteps")
			convey.So(program.Steps[5].Out["loop"], convey.ShouldEqual, "timesteps")
			convey.So(program.Steps[6].Op, convey.ShouldEqual, "control.loop_each")
			convey.So(program.Steps[6].Loop.Over, convey.ShouldEqual, "timesteps")
			convey.So(program.Steps[6].Loop.As, convey.ShouldEqual, "timestep")
			convey.So(program.Steps[6].Body[0].In["hidden_states"], convey.ShouldEqual, "state.latents")
			convey.So(program.Steps[6].Body[0].In["encoder_hidden_states"], convey.ShouldEqual, "state.text_embedding")
			convey.So(program.Steps[6].Body[0].In["timestep"], convey.ShouldEqual, "timestep")
			convey.So(program.Steps[8].Config["path"], convey.ShouldEqual, "flux-2-klein-4b.png")
			convey.So(program.Steps[8].Config["width"], convey.ShouldEqual, 256)
			convey.So(program.Steps[8].Config["height"], convey.ShouldEqual, 256)
		})
	})
}
