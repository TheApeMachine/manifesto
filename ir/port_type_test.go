package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestDimensionIsSymbolicAndString(t *testing.T) {
	convey.Convey("Given a Dimension", t, func() {
		convey.Convey("Symbolic dimensions report IsSymbolic and render the symbol", func() {
			dimension := Dimension{Symbol: "B", Static: 0}

			convey.So(dimension.IsSymbolic(), convey.ShouldBeTrue)
			convey.So(dimension.String(), convey.ShouldEqual, "B")
		})

		convey.Convey("Static dimensions report not IsSymbolic and render the size", func() {
			dimension := Dimension{Symbol: "", Static: 128}

			convey.So(dimension.IsSymbolic(), convey.ShouldBeFalse)
			convey.So(dimension.String(), convey.ShouldEqual, "128")
		})
	})
}

func TestShapeSchemaStringMixedDims(t *testing.T) {
	convey.Convey("Given a ShapeSchema with [B, T, 128]", t, func() {
		shape := ShapeSchema{
			Dimensions: []Dimension{
				{Symbol: "B"},
				{Symbol: "T"},
				{Static: 128},
			},
		}

		convey.Convey("It renders as [B, T, 128]", func() {
			convey.So(shape.String(), convey.ShouldEqual, "[B, T, 128]")
		})
	})
}

func TestLayoutSchemaStringEnumerates(t *testing.T) {
	convey.Convey("Given each LayoutSchema constant", t, func() {
		cases := map[LayoutSchema]string{
			LayoutUnspecified:  "Unspecified",
			LayoutContiguous:   "Contiguous",
			LayoutStrided:      "Strided",
			LayoutTiled:        "Tiled",
			LayoutChannelFirst: "ChannelFirst",
			LayoutChannelLast:  "ChannelLast",
		}

		for layout, want := range cases {
			layout := layout
			want := want

			convey.Convey("It renders "+want+" correctly", func() {
				convey.So(layout.String(), convey.ShouldEqual, want)
			})
		}
	})
}

func TestPortTypeConstructionPreservesAllFields(t *testing.T) {
	convey.Convey("Given a PortType for a hidden state", t, func() {
		portType := PortType{
			DType: dtype.Float32,
			ShapeSchema: ShapeSchema{
				Dimensions: []Dimension{{Symbol: "B"}, {Symbol: "T"}, {Static: 768}},
			},
			Layout: LayoutContiguous,
			Kind:   SemanticHiddenState,
			Constraints: []Constraint{
				DivisibilityConstraint{DimensionIndex: -1, Divisor: 8},
			},
		}

		convey.Convey("All fields are preserved", func() {
			convey.So(portType.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(len(portType.ShapeSchema.Dimensions), convey.ShouldEqual, 3)
			convey.So(portType.Layout, convey.ShouldEqual, LayoutContiguous)
			convey.So(portType.Kind, convey.ShouldEqual, SemanticHiddenState)
			convey.So(len(portType.Constraints), convey.ShouldEqual, 1)
		})
	})
}

func TestConstraintImplementationsRenderReadably(t *testing.T) {
	convey.Convey("Given each Constraint kind", t, func() {
		convey.Convey("DivisibilityConstraint renders as 'dim[I] %% D == 0'", func() {
			constraint := DivisibilityConstraint{DimensionIndex: -1, Divisor: 8}
			convey.So(constraint.String(), convey.ShouldEqual, "dim[-1] % 8 == 0")
		})

		convey.Convey("SymbolEqualityConstraint renders as 'A == B'", func() {
			constraint := SymbolEqualityConstraint{LeftSymbol: "B", RightSymbol: "BatchSize"}
			convey.So(constraint.String(), convey.ShouldEqual, "B == BatchSize")
		})

		convey.Convey("RangeConstraint renders as 'Min <= dim[I] <= Max'", func() {
			constraint := RangeConstraint{DimensionIndex: 0, Min: 1, Max: 64}
			convey.So(constraint.String(), convey.ShouldEqual, "1 <= dim[0] <= 64")
		})
	})
}
