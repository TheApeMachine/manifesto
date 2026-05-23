package diffusion

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestPackLatents(t *testing.T) {
	convey.Convey("Given a 1x2x2x2 latent grid", t, func() {
		values := []float32{1, 2, 3, 4, 5, 6, 7, 8}

		convey.Convey("It should pack into row-major token order", func() {
			packed, err := PackLatents(values, 1, 2, 2, 2)

			convey.So(err, convey.ShouldBeNil)
			convey.So(packed, convey.ShouldResemble, []float32{1, 5, 2, 6, 3, 7, 4, 8})
		})
	})
}
