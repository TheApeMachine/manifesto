package typer

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/types"
)

func TestLookupSpecFallsBackToOperationRegistry(testingObject *testing.T) {
	convey.Convey("Given an op registered only in the YAML operation catalog", testingObject, func() {
		registry, err := types.NewOperationRegistry()
		convey.So(err, convey.ShouldBeNil)

		_, inRegistry := registry.Lookup(types.Op("shape.cast"))
		convey.So(inRegistry, convey.ShouldBeTrue)

		_, inSpecTable := specTable["shape.cast"]
		convey.So(inSpecTable, convey.ShouldBeTrue)

		convey.Convey("LookupSpec resolves ops from the registry when absent from specTable", func() {
			spec, ok := LookupSpec("math.scheduler_delta")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(len(spec.Inputs), convey.ShouldBeGreaterThan, 0)
			convey.So(spec.OutputDeriver, convey.ShouldNotBeNil)
		})
	})
}
