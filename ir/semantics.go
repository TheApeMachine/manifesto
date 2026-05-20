package ir

import (
	"fmt"
	"strconv"

	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/dtype"
)

type Layout string

const (
	LayoutDense       Layout = "dense"
	LayoutRowMajor    Layout = "row_major"
	LayoutColumnMajor Layout = "column_major"
)

type MemoryClass string

const (
	MemoryHost    MemoryClass = "host"
	MemoryDevice  MemoryClass = "device"
	MemoryUnified MemoryClass = "unified"
)

type Effect string

const (
	EffectPure       Effect = "pure"
	EffectRandom     Effect = "random"
	EffectStateRead  Effect = "state_read"
	EffectStateWrite Effect = "state_write"
	EffectCheckpoint Effect = "checkpoint"
	EffectExternalIO Effect = "external_io"
)

type AliasKind string

const (
	AliasAllocates AliasKind = "allocates"
	AliasInput     AliasKind = "input"
	AliasUnknown   AliasKind = "unknown"
)

type Alias struct {
	Kind       AliasKind
	InPlace    bool
	InputIndex int
}

type ValueType struct {
	Shape       tensor.Shape
	DType       dtype.DType
	Layout      Layout
	MemoryClass MemoryClass
	Precision   dtype.DType
}

type AttributeKind string

const (
	AttributeString AttributeKind = "s"
	AttributeInt    AttributeKind = "i"
	AttributeFloat  AttributeKind = "f"
	AttributeBool   AttributeKind = "b"
)

type Attribute struct {
	Kind  AttributeKind
	Value string
}

func StringAttribute(value string) Attribute {
	return Attribute{Kind: AttributeString, Value: value}
}

func IntAttribute(value int64) Attribute {
	return Attribute{Kind: AttributeInt, Value: strconv.FormatInt(value, 10)}
}

func FloatAttribute(value float64) Attribute {
	return Attribute{Kind: AttributeFloat, Value: strconv.FormatFloat(value, 'g', -1, 64)}
}

func BoolAttribute(value bool) Attribute {
	return Attribute{Kind: AttributeBool, Value: strconv.FormatBool(value)}
}

func (attribute Attribute) String() string {
	if attribute.Kind == "" {
		return ""
	}

	return fmt.Sprintf("%s:%s", attribute.Kind, attribute.Value)
}
