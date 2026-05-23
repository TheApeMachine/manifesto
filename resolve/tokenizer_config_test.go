package resolve

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestTokenizerVariables(testingObject *testing.T) {
	convey.Convey("Given a FLUX.2 Klein tokenizer_config.json", testingObject, func() {
		tokenizerConfig := map[string]any{
			"model_max_length": 131072,
			"pad_token":        "<|endoftext|>",
			"eos_token":        "<|im_end|>",
			"added_tokens_decoder": map[string]any{
				"151643": map[string]any{"content": "<|endoftext|>"},
				"151645": map[string]any{"content": "<|im_end|>"},
			},
		}

		convey.Convey("It should derive pad and length fields from repo metadata", func() {
			variables := TokenizerVariables(tokenizerConfig)

			convey.So(variables["model_max_length"], convey.ShouldEqual, 131072)
			convey.So(variables["pad_token_id"], convey.ShouldEqual, 151643)
			convey.So(variables["eos_token_id"], convey.ShouldEqual, 151645)
		})
	})
}
