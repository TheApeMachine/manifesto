package compiler

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"iter"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/types"
)

func TestCompilerBuild(t *testing.T) {
	convey.Convey("Given a topology recipe and safetensors parser", t, func() {
		parser := newTestParser(t, testArchive(t))
		compiler, err := NewCompiler(context.Background(), nil, parser)
		convey.So(err, convey.ShouldBeNil)

		topology := &ast.Topology{
			Nodes: []ast.Node{
				{ID: "x_embedder", Op: "projection.linear"},
				{ID: "transformer_blocks.0.attn.norm_q", Op: "math.rmsnorm"},
			},
		}

		convey.Convey("It should build Project → Architecture → Topology → Node", func() {
			project, err := compiler.WithTopology(topology).Build()
			convey.So(err, convey.ShouldBeNil)
			convey.So(project.Kind, convey.ShouldEqual, ir.KindResearchProject)
			convey.So(project.Metadata["model_type"], convey.ShouldEqual, "flux")
			convey.So(project.Architecture, convey.ShouldNotBeNil)
			convey.So(project.Architecture.Topology, convey.ShouldNotBeNil)
			convey.So(len(project.Architecture.Topology.Nodes), convey.ShouldEqual, 2)

			embedder := project.Node("x_embedder")
			convey.So(embedder, convey.ShouldNotBeNil)
			convey.So(embedder.Kind, convey.ShouldEqual, ir.KindNode)
			convey.So(embedder.Op, convey.ShouldEqual, types.Op("projection.linear"))
			convey.So(embedder.BindMethod, convey.ShouldEqual, "Matmul")
			convey.So(embedder.Weight.HasTensor(), convey.ShouldBeTrue)
			convey.So(embedder.Weight.Tensor.Name, convey.ShouldEqual, "x_embedder.weight")

			norm := project.Node("transformer_blocks.0.attn.norm_q")
			convey.So(norm, convey.ShouldNotBeNil)
			convey.So(norm.Op, convey.ShouldEqual, types.Op("math.rmsnorm"))
			convey.So(norm.BindMethod, convey.ShouldEqual, "RMSNorm")
		})
	})
}

func BenchmarkCompilerBuild(b *testing.B) {
	parser := newTestParser(b, testArchive(b))
	compiler, err := NewCompiler(context.Background(), nil, parser)

	if err != nil {
		b.Fatal(err)
	}

	topology := &ast.Topology{
		Nodes: []ast.Node{
			{ID: "x_embedder", Op: "projection.linear"},
			{ID: "transformer_blocks.0.attn.norm_q", Op: "math.rmsnorm"},
		},
	}

	compiler = compiler.WithTopology(topology)

	for b.Loop() {
		if _, err := compiler.Build(); err != nil {
			b.Fatal(err)
		}
	}
}

type testParser struct {
	archive []byte
}

func (testParser *testParser) Generate() iter.Seq[types.Token] {
	headerLength := binary.LittleEndian.Uint64(testParser.archive[:8])
	headerBytes := testParser.archive[8 : 8+headerLength]

	fields := make(map[string]json.RawMessage)

	if err := json.Unmarshal(headerBytes, &fields); err != nil {
		return func(yield func(types.Token) bool) {}
	}

	tokens := make([]types.Token, 0, len(fields))

	for name, rawField := range fields {
		if name == "__metadata__" {
			var metadata map[string]string

			if err := json.Unmarshal(rawField, &metadata); err != nil {
				return func(yield func(types.Token) bool) {}
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
			return func(yield func(types.Token) bool) {}
		}

		precision, err := dtype.Parse(entry.DType)

		if err != nil {
			return func(yield func(types.Token) bool) {}
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
	}
}

func newTestParser(tb testing.TB, archive []byte) types.Parser {
	tb.Helper()

	return &testParser{archive: archive}
}

func testArchive(tb testing.TB) []byte {
	tb.Helper()

	header := map[string]any{
		"__metadata__": map[string]string{
			"format":     "pt",
			"model_type": "flux",
		},
		"x_embedder.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{128, 3072},
			"data_offsets": []int64{0, 786432},
		},
		"transformer_blocks.0.attn.norm_q.weight": map[string]any{
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
