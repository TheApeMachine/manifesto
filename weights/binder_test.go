package weights

import (
	"iter"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/types"
)

func TestBinderBindAttachesBiasName(testingObject *testing.T) {
	convey.Convey("Given matching weight and bias tensors", testingObject, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID: "decoder.conv",
				},
			},
		}
		parser := tokenParser{
			tokens: []types.Token{
				{
					Kind:      types.KindTensor,
					Name:      "decoder.conv.weight",
					Precision: dtype.Float32,
					Shape:     []int64{4, 3, 3, 3},
				},
				{
					Kind:      types.KindTensor,
					Name:      "decoder.conv.bias",
					Precision: dtype.Float32,
					Shape:     []int64{4},
				},
			},
		}

		convey.Convey("It should preserve the paired bias name", func() {
			used, err := NewBinder().Bind(graph, parser, nil)

			convey.So(err, convey.ShouldBeNil)
			convey.So(graph.Nodes[0].Weights.TensorName, convey.ShouldEqual, "decoder.conv.weight")
			convey.So(graph.Nodes[0].Weights.BiasName, convey.ShouldEqual, "decoder.conv.bias")
			convey.So(used, convey.ShouldContainKey, "decoder.conv.weight")
			convey.So(used, convey.ShouldContainKey, "decoder.conv.bias")
		})
	})
}

type tokenParser struct {
	tokens []types.Token
}

func (parser tokenParser) Generate() (iter.Seq[types.Token], error) {
	return func(yield func(types.Token) bool) {
		for _, token := range parser.tokens {
			if !yield(token) {
				return
			}
		}
	}, nil
}
