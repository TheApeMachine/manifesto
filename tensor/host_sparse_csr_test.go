package tensor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestHostBackend_UploadSparseCSR(t *testing.T) {
	convey.Convey("Given a CSR-encoded 4x4 sparse matrix", t, func() {
		// Logical layout:
		//   1 0 0 2
		//   0 3 0 0
		//   0 0 4 5
		//   6 0 0 0
		// nnz = 6
		// row_ptr = [0, 2, 3, 5, 6]
		// col_idx = [0, 3, 1, 2, 3, 0]
		// values  = [1, 2, 3, 4, 5, 6] (float32)
		backend := NewHostBackend()
		defer backend.Close()

		shape, _ := NewShape([]int{4, 4})

		rowPtrShape, _ := NewShape([]int{5})
		rowPtrBytes := make([]byte, 5*4)
		for index, value := range []uint32{0, 2, 3, 5, 6} {
			binary.LittleEndian.PutUint32(rowPtrBytes[index*4:], value)
		}
		rowPtr, _ := backend.Upload(rowPtrShape, dtype.Int32, rowPtrBytes)
		defer rowPtr.Close()

		colIdxShape, _ := NewShape([]int{6})
		colIdxBytes := make([]byte, 6*4)
		for index, value := range []uint32{0, 3, 1, 2, 3, 0} {
			binary.LittleEndian.PutUint32(colIdxBytes[index*4:], value)
		}
		colIdx, _ := backend.Upload(colIdxShape, dtype.Int32, colIdxBytes)
		defer colIdx.Close()

		valuesBytes := make([]byte, 6*4)
		for index, value := range []float32{1, 2, 3, 4, 5, 6} {
			binary.LittleEndian.PutUint32(valuesBytes[index*4:], math.Float32bits(value))
		}

		convey.Convey("UploadSparse should produce a SparseTensor", func() {
			sparse, err := backend.UploadSparse(
				shape,
				dtype.Float32,
				LayoutSparseCSR,
				valuesBytes,
				[]SparseIndex{
					{Name: "row_ptr", Data: rowPtr},
					{Name: "col_idx", Data: colIdx},
				},
			)

			convey.So(err, convey.ShouldBeNil)
			defer sparse.Close()

			convey.So(sparse.NNZ(), convey.ShouldEqual, 6)
			convey.So(sparse.Layout(), convey.ShouldEqual, LayoutSparseCSR)
			convey.So(sparse.DType(), convey.ShouldEqual, dtype.Float32)
			convey.So(sparse.Shape().Dims(), convey.ShouldResemble, []int{4, 4})

			values, err := sparse.Values()
			convey.So(err, convey.ShouldBeNil)

			view, _ := values.Float32Native()
			convey.So(view, convey.ShouldResemble, []float32{1, 2, 3, 4, 5, 6})

			indices, err := sparse.Indices()
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(indices), convey.ShouldEqual, 2)
			convey.So(indices[0].Name, convey.ShouldEqual, "row_ptr")
			convey.So(indices[1].Name, convey.ShouldEqual, "col_idx")
		})
	})

	convey.Convey("Given an unsupported layout", t, func() {
		backend := NewHostBackend()
		defer backend.Close()

		shape, _ := NewShape([]int{4, 4})

		_, err := backend.UploadSparse(
			shape,
			dtype.Float32,
			LayoutSparseCOO,
			nil,
			nil,
		)

		convey.Convey("It should reject with ErrLayoutUnsupported", func() {
			convey.So(err, convey.ShouldEqual, ErrLayoutUnsupported)
		})
	})
}
