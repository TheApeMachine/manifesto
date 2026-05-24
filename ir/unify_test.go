package ir

import (
	"errors"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func makeShape(dims ...any) ShapeSchema {
	out := ShapeSchema{Dimensions: make([]Dimension, len(dims))}

	for index, raw := range dims {
		switch typed := raw.(type) {
		case string:
			out.Dimensions[index] = Dimension{Symbol: typed}
		case int:
			out.Dimensions[index] = Dimension{Static: int64(typed)}
		case int64:
			out.Dimensions[index] = Dimension{Static: typed}
		}
	}

	return out
}

func makeType(d dtype.DType, shape ShapeSchema, layout LayoutSchema, kind SemanticKind, constraints ...Constraint) PortType {
	return PortType{
		DType:       d,
		ShapeSchema: shape,
		Layout:      layout,
		Kind:        kind,
		Constraints: constraints,
	}
}

func TestUnifyIdenticalTypes(t *testing.T) {
	convey.Convey("Given two identical PortTypes", t, func() {
		portType := makeType(dtype.Float32, makeShape("B", "T", 768), LayoutContiguous, SemanticHiddenState)

		result, err := Unify(portType, portType)

		convey.Convey("Unify succeeds and the result equals the input", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(result.Unified.DType, convey.ShouldEqual, dtype.Float32)
			convey.So(result.Unified.Layout, convey.ShouldEqual, LayoutContiguous)
			convey.So(result.Unified.Kind, convey.ShouldEqual, SemanticHiddenState)
			convey.So(len(result.Bindings), convey.ShouldEqual, 0)
		})
	})
}

func TestUnifyStaticDimsMatch(t *testing.T) {
	convey.Convey("Given two PortTypes with identical static shapes", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 256, 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 256, 768), LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify succeeds with no bindings", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(result.Bindings), convey.ShouldEqual, 0)
		})
	})
}

func TestUnifyStaticDimsDifferRejects(t *testing.T) {
	convey.Convey("Given two PortTypes with conflicting static dims", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 256), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 128), LayoutContiguous, SemanticGeneric)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails with a static-size-mismatch error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "static size mismatch")
		})
	})
}

func TestUnifySymbolBindsToStatic(t *testing.T) {
	convey.Convey("Given producer [B, 768] and consumer [4, 768]", t, func() {
		producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify binds B=4", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(result.Bindings["B"], convey.ShouldEqual, int64(4))
		})

		convey.Convey("The unified dim is the static value", func() {
			convey.So(result.Unified.ShapeSchema.Dimensions[0].IsSymbolic(), convey.ShouldBeFalse)
			convey.So(result.Unified.ShapeSchema.Dimensions[0].Static, convey.ShouldEqual, int64(4))
		})
	})
}

func TestUnifySymbolBindsFromConsumerSide(t *testing.T) {
	convey.Convey("Given producer [4, 768] and consumer [B, 768]", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify binds B=4 from the consumer side", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(result.Bindings["B"], convey.ShouldEqual, int64(4))
		})
	})
}

func TestUnifySameSymbolBothSides(t *testing.T) {
	convey.Convey("Given both shapes use [B, T, 768]", t, func() {
		shape := makeShape("B", "T", 768)
		producer := makeType(dtype.Float32, shape, LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, shape, LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify succeeds with no equality constraints added", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(result.Unified.Constraints), convey.ShouldEqual, 0)
			convey.So(len(result.Bindings), convey.ShouldEqual, 0)
		})
	})
}

func TestUnifyDistinctSymbolsAddsEqualityConstraint(t *testing.T) {
	convey.Convey("Given producer [B, 768] and consumer [N, 768]", t, func() {
		producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape("N", 768), LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify succeeds and adds a SymbolEqualityConstraint(B, N)", func() {
			convey.So(err, convey.ShouldBeNil)

			foundEquality := false
			for _, constraint := range result.Unified.Constraints {
				if eq, ok := constraint.(SymbolEqualityConstraint); ok {
					if eq.LeftSymbol == "B" && eq.RightSymbol == "N" {
						foundEquality = true
					}
				}
			}

			convey.So(foundEquality, convey.ShouldBeTrue)
		})
	})
}

func TestUnifyConflictingSymbolBindingsRejects(t *testing.T) {
	convey.Convey("Given producer [B, B] and consumer [4, 8]", t, func() {
		producer := makeType(dtype.Float32, makeShape("B", "B"), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 8), LayoutContiguous, SemanticGeneric)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails because B cannot bind to both 4 and 8", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "already bound")
		})
	})
}

func TestUnifyRankMismatchRejects(t *testing.T) {
	convey.Convey("Given shapes of different ranks", t, func() {
		producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape("B", "T", 768), LayoutContiguous, SemanticGeneric)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails with a rank-mismatch error", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "rank mismatch")
		})
	})
}

func TestUnifyDTypeMismatchHintsCast(t *testing.T) {
	convey.Convey("Given Float32 producer and Float16 consumer", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float16, makeShape(4, 768), LayoutContiguous, SemanticGeneric)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails and the error carries an AdaptorHint of 'cast'", func() {
			convey.So(err, convey.ShouldNotBeNil)

			var unificationError *UnificationError
			convey.So(errors.As(err, &unificationError), convey.ShouldBeTrue)
			convey.So(unificationError.AdaptorHint, convey.ShouldEqual, "cast")
		})
	})
}

func TestUnifyLayoutMismatchHintsTranspose(t *testing.T) {
	convey.Convey("Given ChannelFirst producer and ChannelLast consumer", t, func() {
		producer := makeType(dtype.Float32, makeShape(1, 3, 224, 224), LayoutChannelFirst, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(1, 3, 224, 224), LayoutChannelLast, SemanticGeneric)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails and the error carries an AdaptorHint of 'transpose'", func() {
			convey.So(err, convey.ShouldNotBeNil)

			var unificationError *UnificationError
			convey.So(errors.As(err, &unificationError), convey.ShouldBeTrue)
			convey.So(unificationError.AdaptorHint, convey.ShouldEqual, "transpose")
		})
	})
}

func TestUnifyLayoutUnspecifiedDefersToOther(t *testing.T) {
	convey.Convey("Given LayoutUnspecified producer", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutUnspecified, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify succeeds with the more specific Layout", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(result.Unified.Layout, convey.ShouldEqual, LayoutContiguous)
		})
	})
}

func TestUnifyKindGenericIsWildcard(t *testing.T) {
	convey.Convey("Given producer Kind=Generic and consumer Kind=HiddenState", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)
		consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticHiddenState)

		result, err := Unify(producer, consumer)

		convey.Convey("Unify succeeds with the more specific Kind", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(result.Unified.Kind, convey.ShouldEqual, SemanticHiddenState)
		})
	})
}

func TestUnifyKindMismatchRejects(t *testing.T) {
	convey.Convey("Given producer Kind=Logits and consumer Kind=HiddenState", t, func() {
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticLogits)
		consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticHiddenState)

		_, err := Unify(producer, consumer)

		convey.Convey("Unify fails because semantic kinds disagree", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "semantic kind mismatch")
		})
	})
}

func TestUnifyDivisibilityConstraintEnforced(t *testing.T) {
	convey.Convey("Given a constraint that the last dim be divisible by 8", t, func() {
		divisibility := DivisibilityConstraint{DimensionIndex: -1, Divisor: 8}

		convey.Convey("A unified dim of 768 satisfies the constraint", func() {
			producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric, divisibility)
			consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric)

			_, err := Unify(producer, consumer)

			convey.So(err, convey.ShouldBeNil)
		})

		convey.Convey("A unified dim of 100 violates the constraint", func() {
			producer := makeType(dtype.Float32, makeShape("B", 100), LayoutContiguous, SemanticGeneric, divisibility)
			consumer := makeType(dtype.Float32, makeShape(4, 100), LayoutContiguous, SemanticGeneric)

			_, err := Unify(producer, consumer)

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "not divisible by 8")
		})
	})
}

func TestUnifyRangeConstraintEnforced(t *testing.T) {
	convey.Convey("Given a constraint that dim[0] be in [1, 64]", t, func() {
		rangeC := RangeConstraint{DimensionIndex: 0, Min: 1, Max: 64}

		convey.Convey("A bound symbol value of 32 satisfies the constraint", func() {
			producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric, rangeC)
			consumer := makeType(dtype.Float32, makeShape(32, 768), LayoutContiguous, SemanticGeneric)

			_, err := Unify(producer, consumer)

			convey.So(err, convey.ShouldBeNil)
		})

		convey.Convey("A bound symbol value of 128 violates the constraint", func() {
			producer := makeType(dtype.Float32, makeShape("B", 768), LayoutContiguous, SemanticGeneric, rangeC)
			consumer := makeType(dtype.Float32, makeShape(128, 768), LayoutContiguous, SemanticGeneric)

			_, err := Unify(producer, consumer)

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "outside [1, 64]")
		})
	})
}

func TestUnifyMergedConstraintsDeduped(t *testing.T) {
	convey.Convey("Given both sides carry the same divisibility constraint", t, func() {
		divisibility := DivisibilityConstraint{DimensionIndex: -1, Divisor: 8}
		producer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric, divisibility)
		consumer := makeType(dtype.Float32, makeShape(4, 768), LayoutContiguous, SemanticGeneric, divisibility)

		result, err := Unify(producer, consumer)

		convey.Convey("The unified constraints list contains exactly one entry", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(result.Unified.Constraints), convey.ShouldEqual, 1)
		})
	})
}

func TestFormatPortTypeRendersConcisely(t *testing.T) {
	convey.Convey("Given a typical hidden-state PortType", t, func() {
		portType := makeType(dtype.Float32, makeShape("B", "T", 768), LayoutContiguous, SemanticHiddenState)

		convey.Convey("formatPortType renders dtype + shape + layout + kind", func() {
			rendered := formatPortType(portType)

			convey.So(rendered, convey.ShouldContainSubstring, "F32")
			convey.So(rendered, convey.ShouldContainSubstring, "[B, T, 768]")
			convey.So(rendered, convey.ShouldContainSubstring, "Contiguous")
			convey.So(rendered, convey.ShouldContainSubstring, "HiddenState")
		})
	})
}
