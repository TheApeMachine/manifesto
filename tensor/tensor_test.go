package tensor

import (
	"testing"
	"unsafe"

	convey "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestNewShape(t *testing.T) {
	convey.Convey("Given valid dimensions", t, func() {
		shape, err := NewShape([]int{2, 3, 4})

		convey.Convey("It should compute the element count", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Valid(), convey.ShouldBeTrue)
			convey.So(shape.Len(), convey.ShouldEqual, 24)
			convey.So(shape.Rank(), convey.ShouldEqual, 3)
		})
	})

	convey.Convey("Given a negative dimension", t, func() {
		shape, err := NewShape([]int{2, -1})

		convey.Convey("It should reject the shape", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(shape.Valid(), convey.ShouldBeFalse)
		})
	})

	convey.Convey("Given a scalar shape", t, func() {
		shape, err := NewShape(nil)

		convey.Convey("It should represent one element", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Len(), convey.ShouldEqual, 1)
		})
	})

	convey.Convey("Given a zero dimension", t, func() {
		shape, err := NewShape([]int{4, 0, 8})

		convey.Convey("It should represent an empty tensor", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Len(), convey.ShouldEqual, 0)
		})
	})
}

func TestShape_Bytes(t *testing.T) {
	convey.Convey("Given a valid shape and dtype", t, func() {
		shape, _ := NewShape([]int{2, 3})

		convey.Convey("It should compute storage bytes per dtype", func() {
			bytesNeeded, err := shape.Bytes(dtype.Float64)
			convey.So(err, convey.ShouldBeNil)
			convey.So(bytesNeeded, convey.ShouldEqual, 48)

			bytesNeeded, err = shape.Bytes(dtype.BFloat16)
			convey.So(err, convey.ShouldBeNil)
			convey.So(bytesNeeded, convey.ShouldEqual, 12)
		})
	})

	convey.Convey("Given packed dtypes", t, func() {
		shape, _ := NewShape([]int{17})

		convey.Convey("Int4 should round up to bytes", func() {
			bytesNeeded, err := shape.Bytes(dtype.Int4)
			convey.So(err, convey.ShouldBeNil)
			convey.So(bytesNeeded, convey.ShouldEqual, 9)
		})

		convey.Convey("Bool should round up to bytes", func() {
			bytesNeeded, err := shape.Bytes(dtype.Bool)
			convey.So(err, convey.ShouldBeNil)
			convey.So(bytesNeeded, convey.ShouldEqual, 3)
		})
	})
}

func TestNewHostBackend(t *testing.T) {
	convey.Convey("Given a fresh host backend", t, func() {
		backend := NewHostBackend()
		defer func() { convey.So(backend.Close(), convey.ShouldBeNil) }()

		convey.Convey("It should report host location and dense layout support", func() {
			convey.So(backend.Location(), convey.ShouldEqual, Host)
			convey.So(backend.SupportedLayouts(), convey.ShouldContain, LayoutDense)
		})
	})
}

func TestHostBackend_Upload_Float32(t *testing.T) {
	convey.Convey("Given float32 bytes", t, func() {
		backend := NewHostBackend()
		defer backend.Close()

		shape, _ := NewShape([]int{4})
		values := []float32{1.0, 2.0, -3.0, 4.25}
		bytesIn := make([]byte, 16)

		for index, value := range values {
			bits := *(*uint32)(unsafe.Pointer(&value))
			bytesIn[index*4+0] = byte(bits)
			bytesIn[index*4+1] = byte(bits >> 8)
			bytesIn[index*4+2] = byte(bits >> 16)
			bytesIn[index*4+3] = byte(bits >> 24)
		}

		convey.Convey("Upload should produce a Float32 host tensor", func() {
			tensor, err := backend.Upload(shape, dtype.Float32, bytesIn)
			convey.So(err, convey.ShouldBeNil)
			defer tensor.Close()

			convey.So(tensor.DType(), convey.ShouldEqual, dtype.Float32)
			convey.So(tensor.Location(), convey.ShouldEqual, Host)
			convey.So(tensor.Layout(), convey.ShouldEqual, LayoutDense)
			convey.So(tensor.Len(), convey.ShouldEqual, 4)
			convey.So(tensor.Bytes(), convey.ShouldEqual, 16)

			view, err := tensor.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(view, convey.ShouldResemble, values)
		})
	})
}

func TestHostBackend_Upload_BFloat16(t *testing.T) {
	convey.Convey("Given bf16 bytes", t, func() {
		backend := NewHostBackend()
		defer backend.Close()

		shape, _ := NewShape([]int{3})
		original := []float32{1.0, -2.0, 0.5}
		bf16s := make([]dtype.BF16, len(original))

		for index, value := range original {
			bf16s[index] = dtype.NewBfloat16FromFloat32(value)
		}

		bytesIn := make([]byte, len(bf16s)*2)

		for index, value := range bf16s {
			bits := value.Bits()
			bytesIn[index*2+0] = byte(bits)
			bytesIn[index*2+1] = byte(bits >> 8)
		}

		convey.Convey("Upload should produce a BFloat16 host tensor", func() {
			tensor, err := backend.Upload(shape, dtype.BFloat16, bytesIn)
			convey.So(err, convey.ShouldBeNil)
			defer tensor.Close()

			convey.So(tensor.DType(), convey.ShouldEqual, dtype.BFloat16)
			convey.So(tensor.Bytes(), convey.ShouldEqual, 6)

			view, err := tensor.BFloat16Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(view), convey.ShouldEqual, 3)

			for index, expected := range bf16s {
				convey.So(view[index], convey.ShouldEqual, expected)
			}
		})
	})
}

func TestHostBackend_Upload_DTypeMismatch(t *testing.T) {
	convey.Convey("Given a Float32 tensor", t, func() {
		backend := NewHostBackend()
		defer backend.Close()

		shape, _ := NewShape([]int{4})
		bytesIn := make([]byte, 16)
		tensor, _ := backend.Upload(shape, dtype.Float32, bytesIn)
		defer tensor.Close()

		convey.Convey("BFloat16Native should reject with ErrDTypeMismatch", func() {
			view, err := tensor.BFloat16Native()
			convey.So(err, convey.ShouldEqual, ErrDTypeMismatch)
			convey.So(view, convey.ShouldBeNil)
		})
	})
}

func TestNew_Uninitialized(t *testing.T) {
	convey.Convey("Given a New tensor", t, func() {
		shape, _ := NewShape([]int{8})
		tensor, err := New(shape, dtype.Float32)

		convey.Convey("It should compile and expose a Float32Native view", func() {
			convey.So(err, convey.ShouldBeNil)
			defer tensor.Close()

			view, err := tensor.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(view), convey.ShouldEqual, 8)
		})
	})
}

func TestNewZeroed(t *testing.T) {
	convey.Convey("Given a NewZeroed tensor", t, func() {
		shape, _ := NewShape([]int{8})
		tensor, err := NewZeroed(shape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)
		defer tensor.Close()

		convey.Convey("Every element should be zero", func() {
			view, err := tensor.Float32Native()
			convey.So(err, convey.ShouldBeNil)

			for _, value := range view {
				convey.So(value, convey.ShouldEqual, float32(0))
			}
		})
	})
}

func TestHostTensor_Close(t *testing.T) {
	convey.Convey("Given a host tensor", t, func() {
		shape, _ := NewShape([]int{4})
		tensor, _ := New(shape, dtype.Float32)

		convey.Convey("Close should be idempotent", func() {
			convey.So(tensor.Close(), convey.ShouldBeNil)
			convey.So(tensor.Close(), convey.ShouldBeNil)
		})

		convey.Convey("After close, native views should error", func() {
			_ = tensor.Close()
			_, err := tensor.Float32Native()
			convey.So(err, convey.ShouldEqual, ErrTensorClosed)
		})
	})
}

func TestAllocate_Alignment(t *testing.T) {
	convey.Convey("Given allocations across tier 1 size classes", t, func() {
		sizes := []int{64, 256, 4096, 65536, 1 << 19}

		convey.Convey("Every allocation should be 64-byte aligned", func() {
			for _, size := range sizes {
				buffer, err := Allocate(size)
				convey.So(err, convey.ShouldBeNil)
				convey.So(len(buffer), convey.ShouldBeGreaterThanOrEqualTo, size)
				convey.So(int(uintptr(unsafe.Pointer(&buffer[0]))%64), convey.ShouldEqual, 0)
				Release(buffer)
			}
		})
	})
}

func TestHostTensor_Slice(t *testing.T) {
	convey.Convey("Given a host tensor", t, func() {
		shape, _ := NewShape([]int{8})
		tensor, _ := NewZeroed(shape, dtype.Float32)
		defer tensor.Close()

		view, _ := tensor.Float32Native()
		for index := range view {
			view[index] = float32(index + 1)
		}

		convey.Convey("Slice should produce a zero-copy subview", func() {
			subview, err := tensor.Slice(2, 4)
			convey.So(err, convey.ShouldBeNil)
			defer subview.Close()

			convey.So(subview.Len(), convey.ShouldEqual, 4)

			sliceView, err := subview.Float32Native()
			convey.So(err, convey.ShouldBeNil)
			convey.So(sliceView, convey.ShouldResemble, []float32{3, 4, 5, 6})
		})
	})
}

func TestArena_BumpAndReset(t *testing.T) {
	convey.Convey("Given a fresh arena", t, func() {
		arena, err := NewArena(64 * 1024)
		convey.So(err, convey.ShouldBeNil)
		defer arena.Close()

		shape, _ := NewShape([]int{16})

		convey.Convey("New should hand out scratch tensors", func() {
			scratch, err := arena.New(shape, dtype.Float32)
			convey.So(err, convey.ShouldBeNil)
			convey.So(scratch.Bytes(), convey.ShouldEqual, 64)
		})

		convey.Convey("Reset should invalidate outstanding handles", func() {
			scratch, _ := arena.New(shape, dtype.Float32)

			arena.Reset()

			_, err := scratch.Float32Native()
			convey.So(err, convey.ShouldEqual, ErrTensorClosed)
		})
	})
}
