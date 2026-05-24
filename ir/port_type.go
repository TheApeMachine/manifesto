package ir

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

/*
PortType is the typed contract on a node's input or output. It carries
the precision, the (possibly symbolic) shape, the memory layout, the
semantic role, and any structural constraints the compiler must enforce
when unifying this port with another.

Defined in ARCHITECTURE.md §4.1. The compiler's Hindley-Milner
unification pass (Phase 2.2) checks that connected ports satisfy
DType ==, ShapeSchema unifies under symbol substitution, Layout ==,
and all Constraints hold.
*/
type PortType struct {
	DType       dtype.DType
	ShapeSchema ShapeSchema
	Layout      LayoutSchema
	Kind        SemanticKind
	Constraints []Constraint
}

/*
ShapeSchema is a list of dimensions, each of which is either a fixed
static size or a named symbol (e.g., "B" for batch, "T" for sequence
length). Symbols allow the same shape to instantiate at different
concrete sizes per execution while remaining typed at compile time.

Example: a transformer hidden-state port has shape [B, T, D] where B
and T are runtime symbols and D is the fixed hidden size.
*/
type ShapeSchema struct {
	Dimensions []Dimension
}

/*
Dimension is a single axis in a shape. Exactly one of Symbol or Static
is meaningful: when Symbol is non-empty, the dimension is dynamic and
its value comes from the SymbolMap at runtime; when Symbol is empty,
Static gives the fixed compile-time size.
*/
type Dimension struct {
	Symbol string
	Static int64
}

/*
IsSymbolic reports whether the dimension is dynamic (resolved at
runtime via a SymbolMap) vs static (known at compile time).
*/
func (dimension Dimension) IsSymbolic() bool {
	return dimension.Symbol != ""
}

/*
String renders the dimension as either the symbol name or the static
size, suitable for diagnostic output.
*/
func (dimension Dimension) String() string {
	if dimension.IsSymbolic() {
		return dimension.Symbol
	}

	return fmt.Sprintf("%d", dimension.Static)
}

/*
String renders the shape schema in [d0, d1, ...] notation.
*/
func (shapeSchema ShapeSchema) String() string {
	parts := make([]string, len(shapeSchema.Dimensions))

	for index, dimension := range shapeSchema.Dimensions {
		parts[index] = dimension.String()
	}

	return "[" + strings.Join(parts, ", ") + "]"
}

/*
LayoutSchema enumerates the supported physical memory layouts. The
compiler must synthesize a layout adaptor (transpose, reshape) when a
producer's Layout differs from a consumer's Layout per §4.2.
*/
type LayoutSchema int

const (
	// LayoutUnspecified means the producer does not constrain the
	// layout and the consumer may pick. Used during early manifest
	// expansion before type inference settles.
	LayoutUnspecified LayoutSchema = iota
	// LayoutContiguous is row-major contiguous in the innermost
	// dimension first. Default for most operations.
	LayoutContiguous
	// LayoutStrided allows non-unit strides between elements.
	LayoutStrided
	// LayoutTiled is blocked-by-cache-tile layout used by cache-aware
	// matmul / conv kernels per §4.4.
	LayoutTiled
	// LayoutChannelFirst is NCHW for vision tensors.
	LayoutChannelFirst
	// LayoutChannelLast is NHWC for vision tensors.
	LayoutChannelLast
)

/*
String returns a human-readable layout name.
*/
func (layoutSchema LayoutSchema) String() string {
	switch layoutSchema {
	case LayoutUnspecified:
		return "Unspecified"
	case LayoutContiguous:
		return "Contiguous"
	case LayoutStrided:
		return "Strided"
	case LayoutTiled:
		return "Tiled"
	case LayoutChannelFirst:
		return "ChannelFirst"
	case LayoutChannelLast:
		return "ChannelLast"
	default:
		return fmt.Sprintf("LayoutSchema(%d)", int(layoutSchema))
	}
}

/*
SemanticKind tags a port with its high-level role in the architecture.
The compiler uses this to:
  - select operation variants that depend on what a tensor means (e.g.,
    Logits get treated differently from HiddenState by samplers)
  - validate that block macros are connected correctly (an
    ActiveInference block expects a BeliefState input, not raw
    embeddings)

String-typed rather than an enum so user-authored manifests can
introduce domain-specific kinds without changing the IR package.
*/
type SemanticKind string

const (
	SemanticGeneric                SemanticKind = ""
	SemanticHiddenState            SemanticKind = "HiddenState"
	SemanticLogits                 SemanticKind = "Logits"
	SemanticEmbedding              SemanticKind = "Embedding"
	SemanticAttentionScore         SemanticKind = "AttentionScore"
	SemanticAdjacencyMatrix        SemanticKind = "AdjacencyMatrix"
	SemanticBeliefState            SemanticKind = "BeliefState"
	SemanticPreferenceDistribution SemanticKind = "PreferenceDistribution"
	SemanticSystemEntropy          SemanticKind = "SystemEntropy"
	SemanticExpectedFreeEnergy     SemanticKind = "ExpectedFreeEnergy"
	SemanticEventTimestamps        SemanticKind = "EventTimestamps"
	SemanticQueryTimestamps        SemanticKind = "QueryTimestamps"
	SemanticIntensityValues        SemanticKind = "IntensityValues"
	SemanticTokenIndex             SemanticKind = "TokenIndex"
	SemanticMask                   SemanticKind = "Mask"
	SemanticNoise                  SemanticKind = "Noise"
)

/*
Constraint is a predicate the compiler must validate when a port type
is instantiated or when two ports unify. Implementations include
divisibility, symbol equality, and range constraints; user code may
add bespoke constraints by implementing the interface.

Constraints are checked after PortType structural unification: two
ports with compatible DType / Shape / Layout / Kind still fail
unification if any constraint on either port is violated by the
unified type.
*/
type Constraint interface {
	// isConstraint is a marker method that keeps the Constraint
	// interface closed to the ir package. New constraint kinds must
	// be defined here so the compiler's verification pass knows how
	// to evaluate them.
	isConstraint()
	String() string
}

/*
DivisibilityConstraint requires a dimension's instantiated size to be
divisible by Divisor. Used for SIMD alignment ("LastDim % 8 == 0") and
tile-friendly shapes.

DimensionIndex is 0-based from the outermost axis; -1 references the
last (innermost) dimension.
*/
type DivisibilityConstraint struct {
	DimensionIndex int
	Divisor        int64
}

func (DivisibilityConstraint) isConstraint() {}

func (constraint DivisibilityConstraint) String() string {
	return fmt.Sprintf("dim[%d] %% %d == 0", constraint.DimensionIndex, constraint.Divisor)
}

/*
SymbolEqualityConstraint requires two dynamic dimension symbols to
resolve to the same concrete size. Used to express "this port's batch
symbol must equal the parent's batch symbol" so the SymbolMap doesn't
end up with B=4 on one port and B=8 on another.
*/
type SymbolEqualityConstraint struct {
	LeftSymbol  string
	RightSymbol string
}

func (SymbolEqualityConstraint) isConstraint() {}

func (constraint SymbolEqualityConstraint) String() string {
	return fmt.Sprintf("%s == %s", constraint.LeftSymbol, constraint.RightSymbol)
}

/*
RangeConstraint requires a dimension's instantiated size to fall within
[Min, Max] inclusive. Used to express attention-head limits, vocab
caps, and other operation-specific bounds.
*/
type RangeConstraint struct {
	DimensionIndex int
	Min            int64
	Max            int64
}

func (RangeConstraint) isConstraint() {}

func (constraint RangeConstraint) String() string {
	return fmt.Sprintf("%d <= dim[%d] <= %d", constraint.Min, constraint.DimensionIndex, constraint.Max)
}
