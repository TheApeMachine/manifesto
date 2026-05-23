package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/diffusion"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestUploadPackedLatents(t *testing.T) {
	convey.Convey("Given manifest layout for 256x256", t, func() {
		layout, err := diffusion.ComputeLatentLayout(256, 256, 16, 128)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should upload packed latents with the expected shape", func() {
			latents, err := uploadPackedLatents(tensor.NewHostBackend(), layout, 1337, dtype.BFloat16)

			convey.So(err, convey.ShouldBeNil)
			convey.So(latents.Shape().Dims(), convey.ShouldResemble, []int{1, 256, 128})
		})
	})
}
