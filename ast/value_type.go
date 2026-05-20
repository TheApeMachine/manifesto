package ast

import "github.com/theapemachine/manifesto/dtype"

/*
Layout describes tensor memory layout for manifest graph values.
*/
type Layout string

const (
	LayoutDense       Layout = "dense"
	LayoutRowMajor    Layout = "row_major"
	LayoutColumnMajor Layout = "column_major"
)

/*
MemoryClass describes where a value is expected to reside during execution.
*/
type MemoryClass string

const (
	MemoryHost   MemoryClass = "host"
	MemoryDevice MemoryClass = "device"
)

/*
ValueType is the manifest graph contract for activation dtypes and precision.
It mirrors compute/ir.ValueType so manifest/ir can lower without reinterpretation.
*/
type ValueType struct {
	Shape     []int64
	DType     dtype.DType
	Layout    Layout
	Memory    MemoryClass
	Precision dtype.DType
}

/*
NewValueType constructs a dense device-resident value type for one execution dtype.
*/
func NewValueType(executionDType dtype.DType) ValueType {
	return ValueType{
		DType:     executionDType,
		Precision: executionDType,
		Layout:    LayoutDense,
		Memory:    MemoryDevice,
	}
}
