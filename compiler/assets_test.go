package compiler

import (
	"context"
	"testing"
	"testing/fstest"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/catalog"
)

func TestCompilerCompileAssetsIncludeObjects(testingObject *testing.T) {
	convey.Convey("Given a program with an included model manifest object", testingObject, func() {
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
outputs:
  - name: sample
    type: tensor
system:
  topology:
    inputs: [hidden_states]
    nodes:
      - id: activation
        op: activation.relu
        in: [hidden_states]
        out: [sample]
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
  - in:
      hidden_states: hidden_states
    op: graph.call
    graph: example
    out:
      sample: sample
`),
		}

		convey.Convey("It should compile the included object as a named graph", func() {
			output, err := compiler.CompileAssets(context.Background(), input, files)

			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Program.IncludeObjects, convey.ShouldContainKey, "example")
			convey.So(output.Graphs, convey.ShouldContainKey, "example")
			convey.So(output.ComputeGraphs, convey.ShouldContainKey, "example")
		})
	})
}

func TestCompilerCompileAssetsTopologyIncludeObjects(testingObject *testing.T) {
	convey.Convey("Given a program that directly includes a topology manifest", testingObject, func() {
		files := fstest.MapFS{
			"model/architecture/registry.yml": {
				Data: []byte("architectures: {}\n"),
			},
			"model/architecture/simple.yml": {
				Data: []byte(`
inputs: [hidden_states]
nodes:
  - id: activation
    op: activation.relu
    in: [hidden_states]
    out: [sample]
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
  simple: model.architecture.simple
main:
  - op: graph.call
    graph: simple
    out:
      sample: sample
`),
		}

		convey.Convey("It should compile the topology manifest as a graph", func() {
			output, err := compiler.CompileAssets(context.Background(), input, files)

			convey.So(err, convey.ShouldBeNil)
			convey.So(output.Graphs, convey.ShouldContainKey, "simple")
			convey.So(output.ComputeGraphs, convey.ShouldContainKey, "simple")
		})
	})
}
