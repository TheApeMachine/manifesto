package parse

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestBlockModel_WeightSubfolder(t *testing.T) {
	convey.Convey("Given a model block with a component weight file", t, func() {
		raw := []byte(`
system:
  runtime:
    model:
      source: black-forest-labs/FLUX.2-klein-4B
      file: text_encoder/model.safetensors.index.json
`)
		block, err := BlockModelFromYAML(raw)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should derive the component subfolder", func() {
			convey.So(block.WeightSubfolder(), convey.ShouldEqual, "text_encoder")
		})
	})
}
