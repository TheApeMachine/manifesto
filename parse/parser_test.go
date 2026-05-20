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
}
