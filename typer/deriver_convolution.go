package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
)

func deriveConv2DOutput(node *ast.GraphNode, inputs []ir.PortType, bindings ir.SymbolMap) (ir.PortType, error) {
	if len(inputs) < 1 {
		return ir.PortType{}, fmt.Errorf("typer: convolution.conv2d needs one input")
	}

	dimensions := inputs[0].ShapeSchema.Dimensions

	if len(dimensions) != 4 {
		return ir.PortType{}, fmt.Errorf("typer: convolution.conv2d input rank must be 4")
	}

	outChannels := conv2DConfigValue(node, "out_channels", boundWeightDim(node, 0))

	if outChannels <= 0 {
		return ir.PortType{}, fmt.Errorf("typer: convolution.conv2d requires positive out_channels")
	}

	inputHeight, err := dimensionInt(dimensions[2], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: convolution.conv2d input height: %w", err)
	}

	inputWidth, err := dimensionInt(dimensions[3], bindings)

	if err != nil {
		return ir.PortType{}, fmt.Errorf("typer: convolution.conv2d input width: %w", err)
	}

	outputHeight, err := conv2DOutputDimension(node, inputHeight, "height")

	if err != nil {
		return ir.PortType{}, err
	}

	outputWidth, err := conv2DOutputDimension(node, inputWidth, "width")

	if err != nil {
		return ir.PortType{}, err
	}

	result := inputs[0]
	result.ShapeSchema = ir.ShapeSchema{
		Dimensions: []ir.Dimension{
			dimensions[0],
			{Static: outChannels},
			{Static: outputHeight},
			{Static: outputWidth},
		},
	}
	result.Kind = ir.SemanticHiddenState

	return result, nil
}

func conv2DOutputDimension(node *ast.GraphNode, inputSize int64, axis string) (int64, error) {
	kernel := conv2DConfigValue(node, "kernel_"+axis[:1], boundConv2DKernelDim(node, axis))
	stride := conv2DConfigValue(node, "stride_"+axis[:1], 1)
	padding := conv2DConfigValue(node, "pad_"+axis[:1], 0)
	dilation := conv2DConfigValue(node, "dil_"+axis[:1], 1)

	if kernel <= 0 {
		return 0, fmt.Errorf("typer: convolution.conv2d %s kernel must be positive", axis)
	}

	if stride <= 0 {
		return 0, fmt.Errorf("typer: convolution.conv2d %s stride must be positive", axis)
	}

	if dilation <= 0 {
		return 0, fmt.Errorf("typer: convolution.conv2d %s dilation must be positive", axis)
	}

	effectiveKernel := dilation*(kernel-1) + 1
	outputSize := (inputSize+2*padding-effectiveKernel)/stride + 1

	if outputSize <= 0 {
		return 0, fmt.Errorf("typer: convolution.conv2d %s output is non-positive", axis)
	}

	return outputSize, nil
}

func conv2DConfigValue(node *ast.GraphNode, key string, defaultValue int64) int64 {
	value := configInt64(node, key)

	if value != 0 {
		return value
	}

	return defaultValue
}

func boundConv2DKernelDim(node *ast.GraphNode, axis string) int64 {
	if axis == "height" {
		return boundWeightDim(node, 2)
	}

	return boundWeightDim(node, 3)
}
