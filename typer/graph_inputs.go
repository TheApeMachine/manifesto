package typer

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
)

/*
graphInputPortType returns a concrete PortType for generic graph boundary
names used by paged-decode manifests. Names outside this set use the
permissive anyTensor() default.
*/
func graphInputPortType(inputName string) (ir.PortType, bool) {
	switch inputName {
	case "input_ids":
		return ir.PortType{
			DType:       dtype.Int32,
			ShapeSchema: shapeSymbols("B", "T"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticTokenIndex,
		}, true
	case "hidden_states", "latents":
		return ir.PortType{
			DType:       dtype.Float32,
			ShapeSchema: shapeSymbols("B", "T", "D"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticHiddenState,
		}, true
	case "encoder_hidden_states", "text_embedding":
		return ir.PortType{
			DType:       dtype.Float32,
			ShapeSchema: shapeSymbols("B", "C", "E"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticHiddenState,
		}, true
	case "timestep":
		return ir.PortType{
			DType:       dtype.Float32,
			ShapeSchema: shapeSymbols("B"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticGeneric,
		}, true
	case "key_pages", "value_pages":
		return ir.PortType{
			DType:       dtype.Float32,
			ShapeSchema: shapeSymbols("L", "P", "S", "KVH", "HD"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticGeneric,
		}, true
	case "write_page_ids", "write_offsets":
		return ir.PortType{
			DType:       dtype.Int32,
			ShapeSchema: shapeSymbols("N"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticGeneric,
		}, true
	case "key_page_table", "value_page_table":
		return ir.PortType{
			DType:       dtype.Int32,
			ShapeSchema: shapeSymbols("P"),
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticGeneric,
		}, true
	case "position_offset":
		return ir.PortType{
			DType:       dtype.Int32,
			ShapeSchema: ir.ShapeSchema{Dimensions: []ir.Dimension{{Static: 1}}},
			Layout:      ir.LayoutContiguous,
			Kind:        ir.SemanticGeneric,
		}, true
	default:
		return ir.PortType{}, false
	}
}
