package runtime

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

func TestBuildPagedPlan(testingObject *testing.T) {
	convey.Convey("Given a prefill batch starting at position zero", testingObject, func() {
		plan, err := BuildPagedPlan(PagedPlanRequest{
			PositionOffset: int32(0),
			TokenIDs:       []int{1, 2, 3, 4, 5},
			PageSize:       16,
		})

		convey.Convey("It should derive contiguous write indices and page table", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(plan.WritePageIDs, convey.ShouldResemble, []int32{0, 0, 0, 0, 0})
			convey.So(plan.WriteOffsets, convey.ShouldResemble, []int32{0, 1, 2, 3, 4})
			convey.So(plan.PageTable, convey.ShouldResemble, []int32{0})
			convey.So(plan.KVLength, convey.ShouldEqual, 5)
		})
	})

	convey.Convey("Given a single-token decode step after prefill", testingObject, func() {
		plan, err := BuildPagedPlan(PagedPlanRequest{
			PositionOffset: int32(5),
			TokenIDs:       []int{42},
			PageSize:       16,
		})

		convey.Convey("It should append at the next absolute slot", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(plan.WritePageIDs, convey.ShouldResemble, []int32{0})
			convey.So(plan.WriteOffsets, convey.ShouldResemble, []int32{5})
			convey.So(plan.PageTable, convey.ShouldResemble, []int32{0})
			convey.So(plan.KVLength, convey.ShouldEqual, 6)
		})
	})
}

func TestScalarInt32ValueAcceptsRawDeviceTensor(testingObject *testing.T) {
	convey.Convey("Given raw bytes from a device-resident int32 scalar", testingObject, func() {
		deviceTensor := newRawDeviceTensor(
			testingObject,
			[]int{1},
			dtype.Int32,
			[]byte{5, 0, 0, 0},
		)

		convey.Convey("It should decode through RawBytes", func() {
			value, err := scalarInt32Value(deviceTensor)

			convey.So(err, convey.ShouldBeNil)
			convey.So(value, convey.ShouldEqual, int32(5))
		})
	})
}

func TestDeriveLaunchBindingsPagedDecode(testingObject *testing.T) {
	convey.Convey("Given one decode token and a non-zero KV cursor", testingObject, func() {
		bindings := DeriveLaunchBindings(&ast.Graph{
			Inputs: []string{"input_ids", "position_offset"},
		}, map[string]any{
			"input_ids":       42,
			"position_offset": int32(5),
		})

		convey.Convey("It should bind N to one and KV to cursor plus one", func() {
			convey.So(bindings["N"], convey.ShouldEqual, int64(1))
			convey.So(bindings["T"], convey.ShouldEqual, int64(1))
			convey.So(bindings["KV"], convey.ShouldEqual, int64(6))
		})
	})
}
