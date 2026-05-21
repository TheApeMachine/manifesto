package compiler

import (
	"context"
	"testing"
	"testing/fstest"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/catalog"
)

func TestCompilerFlattenIncludes(testingObject *testing.T) {
	convey.Convey("Given a program with a named manifest include", testingObject, func() {
		files := fstest.MapFS{
			"model/architecture/registry.yml": {
				Data: []byte("architectures: {}\n"),
			},
			"model/example.yml": {
				Data: []byte(`
kind: Block
name: Example
category: model
inputs:
  - name: hidden_states
    type: tensor
system:
  runtime:
    backend: metal
`),
			},
		}
		compiler, err := NewCompiler(Options{Catalog: catalog.NewFS(files)})
		convey.So(err, convey.ShouldBeNil)

		input := CompileInput{
			ProgramYAML: []byte(`
kind: program
name: demo
category: runtime
include:
  example: model.example
main:
  - op: graph.call
    graph: example
    config:
      backend: example.system.runtime.backend
    out:
      value: value
`),
		}

		convey.Convey("It should replace the include location with the included manifest object", func() {
			flattened, err := compiler.flattenIncludes(context.Background(), input, files)

			convey.So(err, convey.ShouldBeNil)
			convey.So(flattened["include"].(map[string]any)["example"].(map[string]any)["kind"], convey.ShouldEqual, "Block")
		})

		convey.Convey("It should resolve dotted include references inside the program", func() {
			flattened, err := compiler.flattenIncludes(context.Background(), input, files)

			convey.So(err, convey.ShouldBeNil)

			mainSteps := flattened["main"].([]any)
			firstStep := mainSteps[0].(map[string]any)
			config := firstStep["config"].(map[string]any)
			convey.So(config["backend"], convey.ShouldEqual, "metal")
			convey.So(firstStep["graph"], convey.ShouldEqual, "example")
		})
	})
}
