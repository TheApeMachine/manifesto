package resolve

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/manifesto/dtype"
)

func TestResolver_ExecutionDType(t *testing.T) {
	convey.Convey("Given a Hugging Face component config", t, func() {
		resolver := NewResolver(nil)

		convey.Convey("It should parse lowercase dtype fields", func() {
			parsed, err := resolver.ExecutionDType(map[string]any{
				"dtype": "bfloat16",
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(parsed, convey.ShouldEqual, dtype.BFloat16)
		})

		convey.Convey("It should reject torch_dtype auto", func() {
			_, err := resolver.ExecutionDType(map[string]any{
				"torch_dtype": "auto",
			})

			convey.So(err, convey.ShouldNotBeNil)
		})

		convey.Convey("It should default to float32 when no dtype is declared", func() {
			parsed, err := resolver.ExecutionDType(map[string]any{
				"num_layers": 12,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(parsed, convey.ShouldEqual, dtype.Float32)
		})
	})
}
