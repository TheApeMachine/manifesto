package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestStrideFormulaResolveConstantOnly(t *testing.T) {
	convey.Convey("Given a stride formula with only a constant term", t, func() {
		formulas := []StrideFormula{
			{Symbol: "", Multiplier: 128},
		}

		convey.Convey("Resolve returns the constant regardless of SymbolMap", func() {
			emptyMap := SymbolMap{}
			convey.So(emptyMap.Resolve(formulas), convey.ShouldEqual, int64(128))

			populatedMap := SymbolMap{"B": 4, "T": 256}
			convey.So(populatedMap.Resolve(formulas), convey.ShouldEqual, int64(128))
		})
	})
}

func TestStrideFormulaResolveSymbolicTerm(t *testing.T) {
	convey.Convey("Given a stride formula T × 128 for the batch stride of [B, T, 128]", t, func() {
		formulas := []StrideFormula{
			{Symbol: "T", Multiplier: 128},
		}

		convey.Convey("Resolve multiplies the symbol value by the multiplier", func() {
			symbols := SymbolMap{"T": 256}
			convey.So(symbols.Resolve(formulas), convey.ShouldEqual, int64(256*128))
		})

		convey.Convey("Unknown symbols resolve to zero (planner bug indicator)", func() {
			emptyMap := SymbolMap{}
			convey.So(emptyMap.Resolve(formulas), convey.ShouldEqual, int64(0))
		})
	})
}

func TestStrideFormulaResolveSumsMultipleTerms(t *testing.T) {
	convey.Convey("Given a stride formula T × 128 + 4 (constant + symbolic)", t, func() {
		formulas := []StrideFormula{
			{Symbol: "T", Multiplier: 128},
			{Symbol: "", Multiplier: 4},
		}

		symbols := SymbolMap{"T": 10}

		convey.Convey("Resolve sums the terms", func() {
			convey.So(symbols.Resolve(formulas), convey.ShouldEqual, int64(10*128+4))
		})
	})
}

func TestPortAllocationConstructionPreservesAllFields(t *testing.T) {
	convey.Convey("Given a PortAllocation for a hidden-state port", t, func() {
		allocation := PortAllocation{
			PortID:     42,
			BaseOffset: 64,
			StrideExprs: []StrideFormula{
				{Symbol: "T", Multiplier: 768 * 4},
				{Symbol: "", Multiplier: 0},
			},
		}

		convey.Convey("All fields are preserved and stride math resolves", func() {
			convey.So(allocation.PortID, convey.ShouldEqual, int32(42))
			convey.So(allocation.BaseOffset, convey.ShouldEqual, int64(64))
			convey.So(len(allocation.StrideExprs), convey.ShouldEqual, 2)

			symbols := SymbolMap{"T": 256}
			convey.So(symbols.Resolve(allocation.StrideExprs), convey.ShouldEqual, int64(256*768*4))
		})
	})
}
