/*
Package typer implements the manifesto compiler's Phase 2.2 type pass:
graph-level Hindley-Milner unification driven over ast.Graph edges, plus
the Phase 2.3 adaptor-synthesis pass that consumes UnificationErrors and
inserts Cast / Transpose / Reshape nodes.

The pass runs after compiler.LowerTopology produces an ast.Graph and
before optimizer.Run. By the time fusion + codegen see the graph, every
node carries InputTypes / OutputType and graph.Bindings holds the
global SymbolMap.
*/
package typer

import (
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
OpSpec describes the typing contract of one op kind. Inputs is the list of
PortType templates the op accepts in order; Output is the PortType template
the op produces. Both may reference symbolic dimensions which the typer
binds against actual upstream types during inference.

Spec lookups by Op string are intentionally a small Go table rather than
a YAML-driven schema. The YAML schemas under template/operation/ are the
long-term source of truth; the bridge from those to OpSpec is a separate
pass and not on the critical path for this stage to be useful.
*/
type OpSpec struct {
	Inputs        []ir.PortType
	Output        ir.PortType
	WeightTypes   []ir.PortType
	OutputDeriver OutputDeriver
}

/*
OutputDeriver is an optional callback that computes a node's output
PortType from its bound input types and attributes. It's used for ops
where the output shape depends on inputs in non-trivial ways (matmul
inner contraction, embedding lookup widening to [N, hidden], etc.).

When OutputDeriver is nil the typer uses OpSpec.Output verbatim with
symbolic dimensions resolved through Bindings.
*/
type OutputDeriver func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error)

/*
specTable holds the OpSpec for every op kind the typer knows about. The
list mirrors the op coverage of execution.opTable so the typer's view
and the dispatcher's view of "supported ops" stay in sync.
*/
var specTable = map[string]OpSpec{
	// Inputs use the rank-1 "N" wildcard so adoptProducerShapeWhenWildcard
	// substitutes the producer's actual shape at unification time. The
	// output deriver then computes the right concrete shape from node
	// config (out_features, d_model). This avoids the trap of a shared
	// symbolic dimension ("D", "D_in") that the typer cannot reconcile
	// when a model uses linears of different widths — e.g. Llama's
	// gate_proj.in_features == 2048 vs down_proj.in_features ==
	// intermediate_size_half, which would simultaneously try to bind the
	// same symbol to two different values.
	"embedding.token": {
		Inputs: []ir.PortType{
			{DType: dtype.Int32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticTokenIndex},
		},
		WeightTypes: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("V", "D"), Layout: ir.LayoutContiguous, Kind: ir.SemanticEmbedding},
		},
		OutputDeriver: deriveEmbeddingOutput,
	},
	"math.rmsnorm": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
		},
		WeightTypes: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("D"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveNormOutput,
	},
	"math.layernorm": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
		},
		WeightTypes: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("D"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveNormOutput,
	},
	"projection.linear": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
		},
		WeightTypes: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("D_in", "D_out"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveLinearOutput,
	},
	"math.matmul": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("M", "K"), Layout: ir.LayoutContiguous},
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("K", "N"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveMatmulOutput,
	},
	"math.add": binaryElementwiseSpec(),
	"math.sub": binaryElementwiseSpec(),
	"math.mul": binaryElementwiseSpec(),
	"math.div": binaryElementwiseSpec(),

	"activation.relu":    unaryElementwiseSpec(),
	"activation.sigmoid": unaryElementwiseSpec(),
	"activation.tanh":    unaryElementwiseSpec(),
	"activation.gelu":    unaryElementwiseSpec(),
	"activation.swish":   unaryElementwiseSpec(),
	"activation.swiglu": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveSwiGLUOutput,
	},

	"shape.view_as_heads": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveViewAsHeadsOutput,
	},
	"shape.merge_heads": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveMergeHeadsOutput,
	},
	"shape.last_token": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveLastTokenOutput,
	},
	"positional.rope": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticHiddenState),
	},
	"attention.gqa": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
		},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticHiddenState),
	},

	"value.assign": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
	},

	// Adaptors the synthesizer emits. Listed so the typer accepts the
	// nodes it produces on its own second pass (idempotency).
	"shape.cast": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveCastOutput,
	},
	"shape.transpose": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveTransposeOutput,
	},
	"shape.reshape": {
		Inputs:        []ir.PortType{anyTensor()},
		OutputDeriver: deriveReshapeOutput,
	},
}

/*
LookupSpec returns the OpSpec for an ast.GraphNode.Op. The bool is false
when the op is not registered; callers can decide whether to default to
a permissive "any tensor in, same out" rule or to fail the pass.
*/
func LookupSpec(op string) (OpSpec, bool) {
	spec, ok := specTable[strings.TrimSpace(op)]
	return spec, ok
}

/*
RegisterSpec adds or overrides an op spec. Out-of-tree extensions inject
their op contracts this way; in-tree ops live in specTable so they stay
grep-friendly.
*/
func RegisterSpec(op string, spec OpSpec) {
	specTable[op] = spec
}

func shapeSymbols(symbols ...string) ir.ShapeSchema {
	dimensions := make([]ir.Dimension, len(symbols))

	for index, symbol := range symbols {
		dimensions[index] = ir.Dimension{Symbol: symbol}
	}

	return ir.ShapeSchema{Dimensions: dimensions}
}

func anyTensor() ir.PortType {
	return ir.PortType{
		DType:       dtype.Float32,
		ShapeSchema: shapeSymbols("N"),
		Layout:      ir.LayoutUnspecified,
		Kind:        ir.SemanticGeneric,
	}
}

func unaryElementwiseSpec() OpSpec {
	return OpSpec{
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
	}
}

func binaryElementwiseSpec() OpSpec {
	return OpSpec{
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
	}
}
