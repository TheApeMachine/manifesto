package diffusion

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestComputeLatentLayout(t *testing.T) {
	convey.Convey("Given manifest generation fields for 256x256", t, func() {
		layout, err := ComputeLatentLayout(256, 256, 16, 128)

		convey.Convey("It should derive packed token topology", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(layout.ImageSeqLen, convey.ShouldEqual, 256)
			convey.So(layout.VAESpatial, convey.ShouldEqual, 32)
			convey.So(layout.PackedChannels, convey.ShouldEqual, 128)
		})
	})

	convey.Convey("Given manifest generation fields for 1024x1024", t, func() {
		layout, err := ComputeLatentLayout(1024, 1024, 16, 128)

		convey.Convey("It should derive packed token topology", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(layout.ImageSeqLen, convey.ShouldEqual, 4096)
			convey.So(layout.LatentSide, convey.ShouldEqual, 64)
		})
	})
}
