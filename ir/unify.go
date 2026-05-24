package ir

import (
	"fmt"
	"strings"
)

/*
UnificationResult is the outcome of unifying two PortTypes.

When Unify succeeds, Unified holds the resulting PortType (with the most
specific Layout and Kind, and a ShapeSchema whose symbolic dimensions
match across producer and consumer), and Bindings holds any symbol →
concrete-size bindings the unification implied (for example, when a
producer's dynamic dimension "B" meets a consumer's static "4",
Bindings will contain {"B": 4}).

Bindings is always non-nil. Unified is the zero PortType when
unification fails (the caller should branch on the error).
*/
type UnificationResult struct {
	Unified  PortType
	Bindings SymbolMap
}

/*
Unify attempts to unify producer and consumer PortTypes per
ARCHITECTURE.md §4.2 direct unification.

This implementation handles the direct path only: if DType, Layout, or
Shape disagree in ways that would require Transpose / Cast / Reshape
adaptors, Unify returns a UnificationError that names the kind of
adaptor the compiler would need to synthesize. The adaptor synthesis
pass itself is a separate future stage (Phase 2.3 in the
manifesto/codegen plan).

Direct unification rules:
  - DType must be identical. A mismatch always requires an explicit
    Cast adaptor; Unify does not perform implicit casts (ARCHITECTURE.md
    §7 bans them).
  - Layout: LayoutUnspecified on either side is treated as "the other
    side's choice." Otherwise both Layouts must match; a mismatch
    requires an explicit Transpose / layout-conversion adaptor.
  - Shape:
      * Rank must match.
      * For each dimension: two statics must be equal; a static + a
        symbol binds that symbol; two symbols with the same name
        unify trivially; two symbols with different names unify with
        an added SymbolEqualityConstraint.
  - Kind: SemanticGeneric on either side acts as a wildcard; otherwise
    both kinds must match exactly. This intentionally rejects "a port
    that produces Logits being consumed where HiddenState is expected"
    so type-aware op variants (samplers, etc.) cannot be accidentally
    fed the wrong tensor.
  - Constraints: the union of producer and consumer constraints is
    validated against the unified shape + bindings. A divisibility or
    range constraint that would be violated by the bound symbol value
    rejects unification.
*/
func Unify(producer, consumer PortType) (UnificationResult, error) {
	result := UnificationResult{Bindings: make(SymbolMap)}

	if err := unifyDType(producer, consumer); err != nil {
		return result, err
	}

	unifiedLayout, err := unifyLayout(producer.Layout, consumer.Layout)
	if err != nil {
		return result, err
	}

	unifiedShape, shapeConstraints, err := unifyShape(
		producer.ShapeSchema,
		consumer.ShapeSchema,
		result.Bindings,
	)
	if err != nil {
		return result, err
	}

	unifiedKind, err := unifyKind(producer.Kind, consumer.Kind)
	if err != nil {
		return result, err
	}

	mergedConstraints := mergeConstraints(producer.Constraints, consumer.Constraints, shapeConstraints)

	if err := validateConstraints(mergedConstraints, unifiedShape, result.Bindings); err != nil {
		return result, err
	}

	result.Unified = PortType{
		DType:       producer.DType,
		ShapeSchema: unifiedShape,
		Layout:      unifiedLayout,
		Kind:        unifiedKind,
		Constraints: mergedConstraints,
	}

	return result, nil
}

/*
UnificationError describes why two PortTypes did not unify. Field hints
help the future adaptor-synthesis pass figure out what kind of adaptor
node to insert (Cast, Transpose, Reshape) or report a hard compilation
error.
*/
type UnificationError struct {
	// Reason is a human-readable explanation suitable for diagnostics.
	Reason string
	// AdaptorHint, if non-empty, names the kind of adaptor that would
	// resolve this mismatch ("cast", "transpose", "reshape"). Empty
	// when no adaptor would help (e.g., rank mismatch, semantic Kind
	// mismatch, constraint violation).
	AdaptorHint string
}

func (unificationError *UnificationError) Error() string {
	return unificationError.Reason
}

func newError(reason string) *UnificationError {
	return &UnificationError{Reason: reason}
}

func newErrorWithHint(reason, hint string) *UnificationError {
	return &UnificationError{Reason: reason, AdaptorHint: hint}
}

func unifyDType(producer, consumer PortType) error {
	if producer.DType == consumer.DType {
		return nil
	}

	return newErrorWithHint(
		fmt.Sprintf("dtype mismatch: producer %s vs consumer %s", producer.DType, consumer.DType),
		"cast",
	)
}

func unifyLayout(producer, consumer LayoutSchema) (LayoutSchema, error) {
	if producer == LayoutUnspecified {
		return consumer, nil
	}

	if consumer == LayoutUnspecified {
		return producer, nil
	}

	if producer == consumer {
		return producer, nil
	}

	return LayoutUnspecified, newErrorWithHint(
		fmt.Sprintf("layout mismatch: producer %s vs consumer %s", producer, consumer),
		"transpose",
	)
}

/*
unifyShape returns the unified ShapeSchema plus any equality constraints
that arise when two distinct symbols meet. Caller-passed bindings are
mutated in place when static ↔ symbol unification binds a symbol.
*/
func unifyShape(
	producer, consumer ShapeSchema,
	bindings SymbolMap,
) (ShapeSchema, []Constraint, error) {
	if len(producer.Dimensions) != len(consumer.Dimensions) {
		return ShapeSchema{}, nil, newError(fmt.Sprintf(
			"rank mismatch: producer %s vs consumer %s",
			producer.String(), consumer.String(),
		))
	}

	dimensions := make([]Dimension, len(producer.Dimensions))
	var equalityConstraints []Constraint

	for index, producerDim := range producer.Dimensions {
		consumerDim := consumer.Dimensions[index]

		unifiedDim, constraint, err := unifyDimension(producerDim, consumerDim, bindings)
		if err != nil {
			return ShapeSchema{}, nil, fmt.Errorf("dim[%d]: %w", index, err)
		}

		dimensions[index] = unifiedDim

		if constraint != nil {
			equalityConstraints = append(equalityConstraints, constraint)
		}
	}

	return ShapeSchema{Dimensions: dimensions}, equalityConstraints, nil
}

/*
unifyDimension handles the four cases for a single axis. Returns the
unified dimension, an optional symbol-equality constraint (only when
two different symbols meet), and an error when statics disagree or a
previously-bound symbol's value conflicts with this binding.
*/
func unifyDimension(
	producer, consumer Dimension,
	bindings SymbolMap,
) (Dimension, Constraint, error) {
	producerIsSymbolic := producer.IsSymbolic()
	consumerIsSymbolic := consumer.IsSymbolic()

	// Both static.
	if !producerIsSymbolic && !consumerIsSymbolic {
		if producer.Static == consumer.Static {
			return producer, nil, nil
		}

		return Dimension{}, nil, newError(fmt.Sprintf(
			"static size mismatch: producer %d vs consumer %d",
			producer.Static, consumer.Static,
		))
	}

	// Producer symbolic, consumer static: bind producer's symbol.
	if producerIsSymbolic && !consumerIsSymbolic {
		if err := bindSymbol(bindings, producer.Symbol, consumer.Static); err != nil {
			return Dimension{}, nil, err
		}

		return consumer, nil, nil
	}

	// Producer static, consumer symbolic: bind consumer's symbol.
	if !producerIsSymbolic && consumerIsSymbolic {
		if err := bindSymbol(bindings, consumer.Symbol, producer.Static); err != nil {
			return Dimension{}, nil, err
		}

		return producer, nil, nil
	}

	// Both symbolic.
	if producer.Symbol == consumer.Symbol {
		return producer, nil, nil
	}

	// Two distinct symbols. Both unify trivially as a fresh symbol
	// (we keep the producer's name) but emit an equality constraint
	// the planner must enforce when these symbols are bound.
	return producer, SymbolEqualityConstraint{
		LeftSymbol:  producer.Symbol,
		RightSymbol: consumer.Symbol,
	}, nil
}

/*
bindSymbol binds a symbol to a value, rejecting the binding when the
symbol is already bound to a conflicting value. This is what catches
"the producer's B is 4 here but somewhere else B was bound to 8" — a
real type error.
*/
func bindSymbol(bindings SymbolMap, symbol string, value int64) error {
	if existing, alreadyBound := bindings[symbol]; alreadyBound {
		if existing == value {
			return nil
		}

		return newError(fmt.Sprintf(
			"symbol %q already bound to %d, cannot rebind to %d",
			symbol, existing, value,
		))
	}

	bindings[symbol] = value
	return nil
}

func unifyKind(producer, consumer SemanticKind) (SemanticKind, error) {
	if producer == SemanticGeneric {
		return consumer, nil
	}

	if consumer == SemanticGeneric {
		return producer, nil
	}

	if producer == consumer {
		return producer, nil
	}

	return SemanticGeneric, newError(fmt.Sprintf(
		"semantic kind mismatch: producer %q vs consumer %q (use a typed adaptor or change the recipe)",
		string(producer), string(consumer),
	))
}

/*
mergeConstraints concatenates the producer, consumer, and unification-
synthesized constraints, deduplicating exact duplicates by String() so
the unified type doesn't carry redundant entries.
*/
func mergeConstraints(producer, consumer, synthesized []Constraint) []Constraint {
	seen := make(map[string]struct{})
	var merged []Constraint

	add := func(constraint Constraint) {
		key := constraint.String()
		if _, exists := seen[key]; exists {
			return
		}

		seen[key] = struct{}{}
		merged = append(merged, constraint)
	}

	for _, constraint := range producer {
		add(constraint)
	}

	for _, constraint := range consumer {
		add(constraint)
	}

	for _, constraint := range synthesized {
		add(constraint)
	}

	return merged
}

/*
validateConstraints checks every constraint against the unified shape
and the symbol bindings collected so far. Constraints that reference
unbound symbols are NOT errors at this stage — they're checked again
at workspace-allocation time when all symbols have concrete values.

What IS an error here:
  - DivisibilityConstraint on a dimension whose value (static or just-
    bound) is not divisible by the divisor.
  - RangeConstraint on a dimension whose value falls outside [Min, Max].
  - SymbolEqualityConstraint where both symbols are already bound to
    different values.
*/
func validateConstraints(
	constraints []Constraint,
	shape ShapeSchema,
	bindings SymbolMap,
) error {
	for _, constraint := range constraints {
		switch typed := constraint.(type) {
		case DivisibilityConstraint:
			value, ok := resolveDimensionValue(shape, typed.DimensionIndex, bindings)
			if !ok {
				continue
			}

			if typed.Divisor == 0 {
				return newError(fmt.Sprintf("divisibility constraint has zero divisor: %s", typed.String()))
			}

			if value%typed.Divisor != 0 {
				return newError(fmt.Sprintf(
					"constraint violated: dim[%d]=%d is not divisible by %d",
					typed.DimensionIndex, value, typed.Divisor,
				))
			}

		case RangeConstraint:
			value, ok := resolveDimensionValue(shape, typed.DimensionIndex, bindings)
			if !ok {
				continue
			}

			if value < typed.Min || value > typed.Max {
				return newError(fmt.Sprintf(
					"constraint violated: dim[%d]=%d is outside [%d, %d]",
					typed.DimensionIndex, value, typed.Min, typed.Max,
				))
			}

		case SymbolEqualityConstraint:
			leftValue, leftBound := bindings[typed.LeftSymbol]
			rightValue, rightBound := bindings[typed.RightSymbol]

			if leftBound && rightBound && leftValue != rightValue {
				return newError(fmt.Sprintf(
					"constraint violated: %q is bound to %d but %q is bound to %d",
					typed.LeftSymbol, leftValue, typed.RightSymbol, rightValue,
				))
			}
		}
	}

	return nil
}

/*
resolveDimensionValue returns the concrete value for a dimension, given
the shape and the current symbol bindings. Returns (value, true) if
the dimension is static or a bound symbol; (0, false) otherwise.

DimensionIndex of -1 means "last dimension," matching the
ARCHITECTURE.md convention in DivisibilityConstraint examples like
"LastDim % 8 == 0."
*/
func resolveDimensionValue(
	shape ShapeSchema,
	dimensionIndex int,
	bindings SymbolMap,
) (int64, bool) {
	if len(shape.Dimensions) == 0 {
		return 0, false
	}

	resolvedIndex := dimensionIndex

	if resolvedIndex < 0 {
		resolvedIndex = len(shape.Dimensions) + resolvedIndex
	}

	if resolvedIndex < 0 || resolvedIndex >= len(shape.Dimensions) {
		return 0, false
	}

	dimension := shape.Dimensions[resolvedIndex]

	if !dimension.IsSymbolic() {
		return dimension.Static, true
	}

	value, bound := bindings[dimension.Symbol]

	return value, bound
}

/*
formatPortType produces a short single-line rendering of a PortType for
diagnostic messages, e.g. "Float32[B, T, 768] Contiguous HiddenState".
Used by tests and by future error messages.
*/
func formatPortType(portType PortType) string {
	parts := []string{
		portType.DType.String(),
		portType.ShapeSchema.String(),
		portType.Layout.String(),
	}

	if portType.Kind != SemanticGeneric {
		parts = append(parts, string(portType.Kind))
	}

	return strings.Join(parts, " ")
}
