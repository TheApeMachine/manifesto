package ir

/*
PortAllocation is the static memory plan for one port: where it lives
in the workspace and how its strides resolve when dynamic symbols are
bound at runtime. Produced by the planner (ARCHITECTURE.md §5.1) after
liveness analysis and interval-coloring allocation.

The executor reads PortAllocation once per session to pre-materialize
device pointers; it never touches PortAllocation in the dispatch loop.
*/
type PortAllocation struct {
	// PortID matches Port.ID (assigned during planning).
	PortID int32
	// BaseOffset is the static byte offset within the global workspace
	// where this port's tensor begins. Always 64-byte aligned per
	// ARCHITECTURE.md §5.1.
	BaseOffset int64
	// StrideExprs encode the per-dimension stride math as a list of
	// symbolic terms. Computed once at launch time by the symbolic
	// stride solver; the dispatch loop reads the resolved value, not
	// the expression.
	StrideExprs []StrideFormula
	// PortType is the typed contract this allocation satisfies. Stored
	// here so the executor can validate dtype/layout assumptions
	// without round-tripping through the Node graph.
	PortType PortType
}

/*
StrideFormula is one term in a symbolic stride expression. A list of
formulas describes a polynomial in the dimension symbols: the resolved
stride is the sum over all formulas of (Multiplier × symbolValue),
where a Symbol of "" denotes the constant term (its Multiplier
contributes directly).

Example: a [B, T, D] tensor with B and T dynamic, D=128, has
- Stride for D axis: [{Symbol:"", Multiplier:1}] = 1
- Stride for T axis: [{Symbol:"", Multiplier:128}] = 128 (= D)
- Stride for B axis: [{Symbol:"T", Multiplier:128}] = T × 128 (= T × D)
*/
type StrideFormula struct {
	Symbol     string
	Multiplier int64
}

/*
SymbolMap binds dynamic shape symbols to concrete sizes at launch
time. Read-only during the execution loop.
*/
type SymbolMap map[string]int64

/*
Resolve evaluates a list of StrideFormulas against a SymbolMap. Unknown
symbols resolve to 0 (which is a planner bug — every dynamic symbol
referenced in a stride expression must appear in the SymbolMap).
*/
func (symbolMap SymbolMap) Resolve(formulas []StrideFormula) int64 {
	var total int64

	for _, formula := range formulas {
		if formula.Symbol == "" {
			total += formula.Multiplier

			continue
		}

		total += formula.Multiplier * symbolMap[formula.Symbol]
	}

	return total
}
