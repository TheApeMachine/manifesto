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

func TestBlockModel_TopologyAST(t *testing.T) {
	convey.Convey("Given a model block with declared outputs", t, func() {
		raw := []byte(`
outputs:
  - name: logits
system:
  topology:
    inputs:
      - input_ids
    nodes:
      - id: lm_head
        op: projection.linear
        in:
          - input_ids
        out:
          - logits
`)
		block, err := BlockModelFromYAML(raw)
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should preserve output refs for lowering", func() {
			topology, topologyErr := block.TopologyAST()

			convey.So(topologyErr, convey.ShouldBeNil)
			convey.So(topology.Outputs["logits"], convey.ShouldEqual, "logits")
		})
	})
}
