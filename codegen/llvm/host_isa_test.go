package llvm

import (
	"runtime"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestHostISALevel(testingObject *testing.T) {
	convey.Convey("Given host ISA detection", testingObject, func() {
		level := HostISALevel()
		width := HostVectorWidth()
		features := CPUFeaturesForLevel(level)

		convey.So(level.String(), convey.ShouldNotBeBlank)
		convey.So(width, convey.ShouldBeGreaterThan, 0)
		convey.So(features, convey.ShouldNotBeBlank)

		convey.Convey("It should match the host architecture", func() {
			switch runtime.GOARCH {
			case "arm64":
				convey.So(level, convey.ShouldEqual, ISALevelNEON)
				convey.So(width, convey.ShouldEqual, 4)
			case "amd64":
				convey.So(width, convey.ShouldBeGreaterThanOrEqualTo, 4)
			}
		})
	})
}

func TestSupportedISALevelsOnHost(testingObject *testing.T) {
	convey.Convey("Given supported ISA levels", testingObject, func() {
		levels := SupportedISALevelsOnHost()
		convey.So(len(levels), convey.ShouldBeGreaterThan, 0)
		convey.So(levels[0], convey.ShouldEqual, HostISALevel())
	})
}
