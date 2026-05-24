package ir

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestIntervalOverlapsSymmetric(t *testing.T) {
	convey.Convey("Given two intervals", t, func() {
		convey.Convey("Disjoint ranges do not overlap (left before right)", func() {
			left := Interval{Start: 0, End: 3}
			right := Interval{Start: 4, End: 7}

			convey.So(left.Overlaps(right), convey.ShouldBeFalse)
			convey.So(right.Overlaps(left), convey.ShouldBeFalse)
		})

		convey.Convey("Touching ranges (end == start) overlap by inclusive semantics", func() {
			left := Interval{Start: 0, End: 3}
			right := Interval{Start: 3, End: 7}

			convey.So(left.Overlaps(right), convey.ShouldBeTrue)
			convey.So(right.Overlaps(left), convey.ShouldBeTrue)
		})

		convey.Convey("Fully nested ranges overlap", func() {
			outer := Interval{Start: 0, End: 10}
			inner := Interval{Start: 2, End: 5}

			convey.So(outer.Overlaps(inner), convey.ShouldBeTrue)
			convey.So(inner.Overlaps(outer), convey.ShouldBeTrue)
		})

		convey.Convey("Partially overlapping ranges overlap", func() {
			left := Interval{Start: 0, End: 5}
			right := Interval{Start: 3, End: 7}

			convey.So(left.Overlaps(right), convey.ShouldBeTrue)
			convey.So(right.Overlaps(left), convey.ShouldBeTrue)
		})
	})
}

func TestWorkspaceLayoutHoldsAllocations(t *testing.T) {
	convey.Convey("Given a WorkspaceLayout with two non-overlapping intervals", t, func() {
		layout := WorkspaceLayout{
			Size:  256,
			Align: 64,
			Allocations: []Interval{
				{PortID: 1, Start: 0, End: 2, Offset: 0, Size: 128},
				{PortID: 2, Start: 3, End: 5, Offset: 0, Size: 128},
			},
		}

		convey.Convey("Coloring allocator may share Offset when intervals are disjoint", func() {
			convey.So(layout.Allocations[0].Overlaps(layout.Allocations[1]), convey.ShouldBeFalse)
			convey.So(layout.Allocations[0].Offset, convey.ShouldEqual, layout.Allocations[1].Offset)
		})

		convey.Convey("All fields are preserved", func() {
			convey.So(layout.Size, convey.ShouldEqual, int64(256))
			convey.So(layout.Align, convey.ShouldEqual, int64(64))
			convey.So(len(layout.Allocations), convey.ShouldEqual, 2)
		})
	})
}
