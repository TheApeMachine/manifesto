package runtime

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewOrchestrator(test *testing.T) {
	Convey("Given orchestrator options", test, func() {
		Convey("It should require hub, compute, and host", func() {
			_, err := NewOrchestrator(OrchestratorOptions{})
			So(err, ShouldNotBeNil)
		})
	})
}
