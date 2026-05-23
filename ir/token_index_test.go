package ir

import (
	"encoding/binary"
	"encoding/json"
	"iter"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/types"
)

func TestNewTokenIndex(t *testing.T) {
	convey.Convey("Given a parser over a minimal safetensors archive", t, func() {
		parser := newArchiveParser(t, minimalArchive(t))
		tokenIndex, err := NewTokenIndex(parser)

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should index tensors and metadata by name", func() {
			convey.So(tokenIndex.TensorCount(), convey.ShouldEqual, 2)
			convey.So(tokenIndex.MetadataCount(), convey.ShouldEqual, 1)

			format, ok := tokenIndex.Metadata("format")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(format.Value, convey.ShouldEqual, "pt")

			weight, ok := tokenIndex.Tensor("x_embedder.weight")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(weight.Shape, convey.ShouldResemble, []int64{128, 3072})
			convey.So(weight.Precision, convey.ShouldEqual, dtype.BFloat16)

			bias, ok := tokenIndex.Tensor("x_embedder.bias")
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(bias.Shape, convey.ShouldResemble, []int64{3072})
		})
	})
}

func TestTokenIndexTensor(t *testing.T) {
	convey.Convey("Given an empty token index", t, func() {
		tokenIndex := &TokenIndex{}

		convey.Convey("It should report missing tensors", func() {
			_, ok := tokenIndex.Tensor("missing")
			convey.So(ok, convey.ShouldBeFalse)
		})
	})
}

func BenchmarkNewTokenIndex(b *testing.B) {
	parser := newArchiveParser(b, minimalArchive(b))

	for b.Loop() {
		_, err := NewTokenIndex(parser)

		if err != nil {
			b.Fatal(err)
		}
	}
}

type archiveParser struct {
	archive []byte
}

func (archiveParser *archiveParser) Generate() (iter.Seq[types.Token], error) {
	headerLength := binary.LittleEndian.Uint64(archiveParser.archive[:8])
	headerBytes := archiveParser.archive[8 : 8+headerLength]

	fields := make(map[string]json.RawMessage)

	if err := json.Unmarshal(headerBytes, &fields); err != nil {
		return nil, err
	}

	tokens := make([]types.Token, 0, len(fields))

	for name, rawField := range fields {
		if name == "__metadata__" {
			var metadata map[string]string

			if err := json.Unmarshal(rawField, &metadata); err != nil {
				return nil, err
			}

			for key, value := range metadata {
				tokens = append(tokens, types.Token{
					Kind:  types.KindMetadata,
					Name:  key,
					Value: value,
				})
			}

			continue
		}

		var entry struct {
			DType       string   `json:"dtype"`
			Shape       []int64  `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}

		if err := json.Unmarshal(rawField, &entry); err != nil {
			return nil, err
		}

		precision, err := dtype.Parse(entry.DType)

		if err != nil {
			return nil, err
		}

		tokens = append(tokens, types.Token{
			Kind:      types.KindTensor,
			Name:      name,
			Shape:     append([]int64(nil), entry.Shape...),
			Precision: precision,
			Span: types.Span{
				Offset: entry.DataOffsets[0],
				Length: entry.DataOffsets[1] - entry.DataOffsets[0],
			},
		})
	}

	return func(yield func(types.Token) bool) {
		for _, token := range tokens {
			if !yield(token) {
				return
			}
		}
	}, nil
}

func newArchiveParser(tb testing.TB, archive []byte) types.Parser {
	tb.Helper()

	return &archiveParser{archive: archive}
}

func minimalArchive(tb testing.TB) []byte {
	tb.Helper()

	header := map[string]any{
		"__metadata__": map[string]string{
			"format": "pt",
		},
		"x_embedder.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{128, 3072},
			"data_offsets": []int64{0, 786432},
		},
		"x_embedder.bias": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{3072},
			"data_offsets": []int64{786432, 792576},
		},
	}

	headerBytes, err := json.Marshal(header)

	if err != nil {
		tb.Fatal(err)
	}

	archive := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(archive[:8], uint64(len(headerBytes)))
	copy(archive[8:], headerBytes)

	return archive
}
