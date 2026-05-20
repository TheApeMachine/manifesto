/*
Package dtype is the canonical source of truth for every numeric format
the platform handles. No other package defines a dtype enum; no other
package hard-codes a numeric format. Conversions to and from these
formats live in pkg/dtype/convert (correctness-only scalar paths) and
pkg/backend/compute/convert (five-host-ISA SIMD paths).

Wire byte order is little-endian everywhere. Safetensors, GGUF, and
every supported target are little-endian; the dtype package matches.
*/
package dtype

import (
	"fmt"
	"strings"
)

/*
DType identifies a scalar storage format. Values are stable; ordering is
load-bearing for snapshot/restore wire formats (§2.10 of
TENSOR_BACKEND_REWRITE.md). Do not reorder existing constants.
*/
type DType uint8

const (
	Invalid DType = iota

	// Floating point.
	Float64
	Float32
	Float16
	BFloat16
	Float8E4M3
	Float8E5M2

	// Signed integers.
	Int64
	Int32
	Int16
	Int8
	Int4 // packed two-per-byte, little-endian nibble order

	// Unsigned integers.
	Uint64
	Uint32
	Uint16
	Uint8

	// Boolean. Packed eight-per-byte, little-endian bit order.
	Bool

	// Complex.
	Complex64
	Complex128
)

/*
Size returns the storage footprint of a single logical element in bytes
for unpacked dtypes. For Int4 it returns 0 with the convention that
callers use BytesFor instead; for Bool it returns 0 likewise. Use
BytesFor(dtype, elementCount) to compute total storage.
*/
func (dtype DType) Size() (int, error) {
	switch dtype {
	case Float64, Int64, Uint64, Complex64:
		return 8, nil
	case Float32, Int32, Uint32:
		return 4, nil
	case Float16, BFloat16, Int16, Uint16:
		return 2, nil
	case Float8E4M3, Float8E5M2, Int8, Uint8:
		return 1, nil
	case Complex128:
		return 16, nil
	case Int4, Bool:
		return 0, nil
	default:
		return 0, fmt.Errorf("dtype: unsupported dtype %d", dtype)
	}
}

/*
BytesFor returns the total storage in bytes for the given number of
logical elements. Handles packed formats (Int4 packs 2 per byte, Bool
packs 8 per byte) correctly.
*/
func (dtype DType) BytesFor(elements int) (int, error) {
	if elements < 0 {
		return 0, fmt.Errorf("dtype: negative element count %d", elements)
	}

	switch dtype {
	case Int4:
		return (elements + 1) / 2, nil
	case Bool:
		return (elements + 7) / 8, nil
	}

	size, err := dtype.Size()

	if err != nil {
		return 0, err
	}

	return elements * size, nil
}

/*
LogicalElements inverts BytesFor: given a byte count, returns the
maximum number of logical elements that fit. For unpacked dtypes,
bytes must be a multiple of Size(); otherwise the function rejects.
For packed dtypes it floors to the nearest packed-element boundary.
*/
func (dtype DType) LogicalElements(bytes int) (int, error) {
	if bytes < 0 {
		return 0, fmt.Errorf("dtype: negative byte count %d", bytes)
	}

	switch dtype {
	case Int4:
		return bytes * 2, nil
	case Bool:
		return bytes * 8, nil
	}

	size, err := dtype.Size()

	if err != nil {
		return 0, err
	}

	if bytes%size != 0 {
		return 0, fmt.Errorf(
			"dtype: byte count %d is not a multiple of dtype size %d",
			bytes, size,
		)
	}

	return bytes / size, nil
}

/*
Name returns the canonical lowercase identifier (e.g. "bf16", "fp8e4m3").
This is the form used in YAML manifests and configuration.
*/
func (dtype DType) Name() string {
	switch dtype {
	case Float64:
		return "f64"
	case Float32:
		return "f32"
	case Float16:
		return "f16"
	case BFloat16:
		return "bf16"
	case Float8E4M3:
		return "fp8e4m3"
	case Float8E5M2:
		return "fp8e5m2"
	case Int64:
		return "i64"
	case Int32:
		return "i32"
	case Int16:
		return "i16"
	case Int8:
		return "i8"
	case Int4:
		return "i4"
	case Uint64:
		return "u64"
	case Uint32:
		return "u32"
	case Uint16:
		return "u16"
	case Uint8:
		return "u8"
	case Bool:
		return "bool"
	case Complex64:
		return "c64"
	case Complex128:
		return "c128"
	case Invalid:
		return "invalid"
	}

	return fmt.Sprintf("dtype(%d)", dtype)
}

/*
String returns the uppercase wire identifier matching safetensors
casing (F32, BF16, I8, BOOL). This is the form used in serialized
formats. For human-readable output prefer Name.
*/
func (dtype DType) String() string {
	switch dtype {
	case Float64:
		return "F64"
	case Float32:
		return "F32"
	case Float16:
		return "F16"
	case BFloat16:
		return "BF16"
	case Float8E4M3:
		return "F8E4M3"
	case Float8E5M2:
		return "F8E5M2"
	case Int64:
		return "I64"
	case Int32:
		return "I32"
	case Int16:
		return "I16"
	case Int8:
		return "I8"
	case Int4:
		return "I4"
	case Uint64:
		return "U64"
	case Uint32:
		return "U32"
	case Uint16:
		return "U16"
	case Uint8:
		return "U8"
	case Bool:
		return "BOOL"
	case Complex64:
		return "C64"
	case Complex128:
		return "C128"
	case Invalid:
		return "INVALID"
	}

	return fmt.Sprintf("DTYPE(%d)", dtype)
}

/*
Parse accepts safetensors-style uppercase strings ("F16", "BF16",
"I8", "BOOL"), the canonical lowercase Name forms ("f16", "bf16",
"i8", "bool"), and a small set of historical aliases ("F8_E4M3",
"FP8_E4M3" → Float8E4M3) used by ecosystems Caramba interoperates
with. Returns Invalid + non-nil error on unknown input.
*/
func Parse(name string) (DType, error) {
	normalized := strings.ToUpper(strings.TrimSpace(name))
	normalized = strings.ReplaceAll(normalized, "_", "")
	normalized = strings.ReplaceAll(normalized, "-", "")

	switch normalized {
	case "F64", "FLOAT64", "DOUBLE":
		return Float64, nil
	case "F32", "FLOAT32", "FLOAT", "SINGLE":
		return Float32, nil
	case "F16", "FLOAT16", "HALF":
		return Float16, nil
	case "BF16", "BFLOAT16", "BRAINFLOAT16":
		return BFloat16, nil
	case "F8E4M3", "FP8E4M3", "FLOAT8E4M3":
		return Float8E4M3, nil
	case "F8E5M2", "FP8E5M2", "FLOAT8E5M2":
		return Float8E5M2, nil
	case "I64", "INT64", "LONG":
		return Int64, nil
	case "I32", "INT32", "INT":
		return Int32, nil
	case "I16", "INT16", "SHORT":
		return Int16, nil
	case "I8", "INT8", "BYTE":
		return Int8, nil
	case "I4", "INT4":
		return Int4, nil
	case "U64", "UINT64", "ULONG":
		return Uint64, nil
	case "U32", "UINT32", "UINT":
		return Uint32, nil
	case "U16", "UINT16", "USHORT":
		return Uint16, nil
	case "U8", "UINT8", "UBYTE":
		return Uint8, nil
	case "BOOL", "BOOLEAN", "B1":
		return Bool, nil
	case "C64", "COMPLEX64":
		return Complex64, nil
	case "C128", "COMPLEX128":
		return Complex128, nil
	}

	return Invalid, fmt.Errorf("dtype: unknown dtype name %q", name)
}

/*
IsFloat reports whether the dtype is a floating-point format
(including BF16 and the FP8 variants).
*/
func (dtype DType) IsFloat() bool {
	switch dtype {
	case Float64, Float32, Float16, BFloat16, Float8E4M3, Float8E5M2:
		return true
	}

	return false
}

/*
IsSignedInt reports whether the dtype is a signed integer.
*/
func (dtype DType) IsSignedInt() bool {
	switch dtype {
	case Int64, Int32, Int16, Int8, Int4:
		return true
	}

	return false
}

/*
IsUnsignedInt reports whether the dtype is an unsigned integer.
*/
func (dtype DType) IsUnsignedInt() bool {
	switch dtype {
	case Uint64, Uint32, Uint16, Uint8:
		return true
	}

	return false
}

/*
IsComplex reports whether the dtype is a complex number.
*/
func (dtype DType) IsComplex() bool {
	switch dtype {
	case Complex64, Complex128:
		return true
	}

	return false
}

/*
IsPacked reports whether the dtype stores multiple logical elements
per byte. Currently Int4 (two per byte) and Bool (eight per byte).
*/
func (dtype DType) IsPacked() bool {
	switch dtype {
	case Int4, Bool:
		return true
	}

	return false
}
