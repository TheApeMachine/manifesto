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
	"fmt"
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
	Inputs       []ir.PortType
	Output       ir.PortType
	WeightTypes  []ir.PortType
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
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticHiddenState),
	},
	"math.layernorm": {
		Inputs: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("N"), Layout: ir.LayoutContiguous, Kind: ir.SemanticHiddenState},
		},
		WeightTypes: []ir.PortType{
			{DType: dtype.Float32, ShapeSchema: shapeSymbols("D"), Layout: ir.LayoutContiguous},
		},
		OutputDeriver: deriveSameAsFirstInput(ir.SemanticHiddenState),
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

/*
deriveSameAsFirstInput returns the first input's PortType verbatim, with
the Kind overridden if a non-generic kind is requested. Used by every
shape-preserving op (norms, activations, elementwise binary).
*/
func deriveSameAsFirstInput(kind ir.SemanticKind) OutputDeriver {
	return func(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
		_ = node
		_ = bindings

		if len(inputs) == 0 {
			return ir.PortType{}, fmt.Errorf("typer: %q has no inputs", node.Op)
		}

		result := inputs[0]

		if kind != ir.SemanticGeneric {
			result.Kind = kind
		}

		return result, nil
	}
}

/*
deriveEmbeddingOutput materializes [N, hidden] from a [N] token-index
input. The hidden dimension comes from the node's `d_model` config
attribute — that's where the architecture template plants the concrete
size, e.g. 2048 for Llama-3.2-1B. Reading it directly produces a static
output dimension so downstream consumers' [N, D] unify against a real
number instead of an unbound symbol the planner then can't size.
*/
func deriveEmbeddingOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: embedding.token needs one input")
	}

	hiddenSize := configInt64(node, "d_model")

	if hiddenSize == 0 {
		hiddenSize = configInt64(node, "hidden_size")
	}

	if hiddenSize == 0 {
		return ir.PortType{}, fmt.Errorf(
			"typer: embedding.token %q requires a d_model (or hidden_size) config attribute",
			node.ID,
		)
	}

	tokenDim := inputs[0].ShapeSchema.Dimensions
	hiddenDim := ir.Dimension{Static: hiddenSize}

	return ir.PortType{
		DType: dtype.Float32,
		ShapeSchema: ir.ShapeSchema{
			Dimensions: append(append([]ir.Dimension(nil), tokenDim...), hiddenDim),
		},
		Layout: ir.LayoutContiguous,
		Kind:   ir.SemanticHiddenState,
	}, nil
}

/*
deriveLinearOutput resolves [N, in_features] × [in_features, out_features]
→ [N, out_features]. The output dimension comes from the node's
`out_features` config attribute — every projection.linear instance in a
HuggingFace-style architecture template has it baked in (q_proj's
out_features is hidden_size; k_proj/v_proj's is num_kv_heads * head_dim;
etc.). Reading it directly keeps each linear's output a unique static
size so the planner can size every intermediate distinctly.

Using a shared "D_out" symbol — the previous implementation's
behaviour — was a bug: Llama's three attention projections produce
different output sizes (2048, 512, 512) yet would unify against the
same symbol, which the typer can't reconcile.
*/
func deriveLinearOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: projection.linear needs one input")
	}

	leading := inputs[0].ShapeSchema.Dimensions

	if len(leading) == 0 {
		return ir.PortType{}, fmt.Errorf("typer: projection.linear input has rank 0")
	}

	outFeatures := configInt64(node, "out_features")

	if outFeatures == 0 {
		return ir.PortType{}, fmt.Errorf(
			"typer: projection.linear %q requires an out_features config attribute",
			node.ID,
		)
	}

	prefix := append([]ir.Dimension(nil), leading[:len(leading)-1]...)
	outDim := ir.Dimension{Static: outFeatures}

	return ir.PortType{
		DType:       dtype.Float32,
		ShapeSchema: ir.ShapeSchema{Dimensions: append(prefix, outDim)},
		Layout:      ir.LayoutContiguous,
		Kind:        ir.SemanticHiddenState,
	}, nil
}

/*
deriveMatmulOutput resolves [M, K] × [K, N] → [M, N].
*/
func deriveMatmulOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: math.matmul needs two inputs")
	}

	leftDims := inputs[0].ShapeSchema.Dimensions
	rightDims := inputs[1].ShapeSchema.Dimensions

	if len(leftDims) < 2 || len(rightDims) < 2 {
		return ir.PortType{}, fmt.Errorf("typer: math.matmul requires rank-2 operands")
	}

	rows := leftDims[len(leftDims)-2]
	cols := rightDims[len(rightDims)-1]

	return ir.PortType{
		DType:       inputs[0].DType,
		ShapeSchema: ir.ShapeSchema{Dimensions: []ir.Dimension{rows, cols}},
		Layout:      ir.LayoutContiguous,
		Kind:        ir.SemanticGeneric,
	}, nil
}

func deriveCastOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.cast needs one input")
	}

	target, _ := node.Attributes["dtype"].(dtype.DType)

	if target == dtype.Invalid {
		target = inputs[0].DType
	}

	result := inputs[0]
	result.DType = target

	return result, nil
}

func deriveTransposeOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.transpose needs one input")
	}

	targetLayout, _ := node.Attributes["layout"].(ir.LayoutSchema)
	result := inputs[0]

	if targetLayout != ir.LayoutUnspecified {
		result.Layout = targetLayout
	}

	return result, nil
}

func deriveReshapeOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	_ = bindings

	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: shape.reshape needs one input")
	}

	dims, _ := node.Attributes["shape"].(ir.ShapeSchema)
	result := inputs[0]

	if len(dims.Dimensions) > 0 {
		result.ShapeSchema = dims
	}

	return result, nil
}

/*
configInt64 reads a node config attribute as an int64. YAML parses
unquoted integer values as int, but template substitution and other
paths sometimes deliver int64 / float64 / json.Number, so the helper
accepts any of those and returns 0 (the architecture template's
"missing" sentinel) when the attribute is absent or non-numeric.

Used by output derivers that need the concrete output dimension a
HuggingFace-style config baked into the node (out_features, d_model,
intermediate_size, num_heads * head_dim, …). Picking the value out of
config here keeps the typer specs free of architecture-specific
symbolic dimensions that would otherwise stay unbound past the planner.
*/
func configInt64(node *ast.GraphNode, key string) int64 {
	if node == nil || node.Attributes == nil {
		return 0
	}

	raw, ok := node.Attributes[key]

	if !ok {
		return 0
	}

	switch typed := raw.(type) {
	case int:
		return int64(typed)
	case int32:
		return int64(typed)
	case int64:
		return typed
	case uint:
		return int64(typed)
	case uint32:
		return int64(typed)
	case uint64:
		return int64(typed)
	case float32:
		return int64(typed)
	case float64:
		return int64(typed)
	}

	return 0
}
