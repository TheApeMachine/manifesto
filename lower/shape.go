package lower

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
)

/*
ShapeInferencer derives activation shapes for manifest topology wires.
*/
type ShapeInferencer struct {
	wires map[string][]int64
}

/*
NewShapeInferencer constructs a ShapeInferencer.
*/
func NewShapeInferencer() *ShapeInferencer {
	return &ShapeInferencer{
		wires: make(map[string][]int64),
	}
}

/*
Apply resolves wire shapes for one lowered graph using topology connectivity.
*/
func (inferencer *ShapeInferencer) Apply(topology *ast.Topology, graph *ast.Graph) error {
	if topology == nil || graph == nil {
		return fmt.Errorf("shape infer: topology and graph are required")
	}

	inferencer.seedInputs(topology.Inputs)

	for nodeIndex, topologyNode := range topology.Nodes {
		inputShapes, err := inferencer.inputShapes(topologyNode.In)

		if err != nil {
			return fmt.Errorf("shape infer node %q: %w", topologyNode.ID, err)
		}

		outputShapes, err := inferencer.inferNode(topologyNode, inputShapes)

		if err != nil {
			return fmt.Errorf("shape infer node %q: %w", topologyNode.ID, err)
		}

		if err := inferencer.publishOutputs(topologyNode.Out, outputShapes); err != nil {
			return fmt.Errorf("shape infer node %q: %w", topologyNode.ID, err)
		}

		if nodeIndex >= len(graph.Nodes) {
			return fmt.Errorf("shape infer node %q: graph node missing", topologyNode.ID)
		}

		graph.Nodes[nodeIndex].ValueType.Shape = cloneShape(outputShapes[0])
	}

	return nil
}

func (inferencer *ShapeInferencer) seedInputs(inputs []string) {
	for _, inputName := range inputs {
		inferencer.wires[inputName] = inferencer.defaultInputShape(inputName)
	}
}

func (inferencer *ShapeInferencer) defaultInputShape(inputName string) []int64 {
	switch {
	case strings.Contains(inputName, "input_ids"),
		strings.Contains(inputName, "position_ids"),
		strings.Contains(inputName, "attention_mask"):
		return ast.NewDynamicShape(2)
	case strings.Contains(inputName, "hidden_states"),
		strings.Contains(inputName, "encoder_hidden_states"),
		strings.Contains(inputName, "latents"):
		return ast.NewDynamicShape(3)
	case strings.Contains(inputName, "timestep"):
		return []int64{ast.DynamicDim}
	default:
		return ast.NewDynamicShape(1)
	}
}

func (inferencer *ShapeInferencer) inputShapes(inputNames []string) ([][]int64, error) {
	shapes := make([][]int64, 0, len(inputNames))

	for _, inputName := range inputNames {
		shape, ok := inferencer.wires[inputName]

		if !ok {
			return nil, fmt.Errorf("unknown wire %q", inputName)
		}

		shapes = append(shapes, cloneShape(shape))
	}

	return shapes, nil
}

func (inferencer *ShapeInferencer) publishOutputs(outputNames []string, shapes [][]int64) error {
	if len(outputNames) == 0 {
		return nil
	}

	if len(outputNames) != len(shapes) {
		return fmt.Errorf("output wire count %d != shape count %d", len(outputNames), len(shapes))
	}

	for index, outputName := range outputNames {
		inferencer.wires[outputName] = cloneShape(shapes[index])
	}

	return nil
}

func (inferencer *ShapeInferencer) inferNode(
	topologyNode ast.Node,
	inputShapes [][]int64,
) ([][]int64, error) {
	switch topologyNode.Op {
	case "embedding.token":
		return inferencer.inferEmbedding(topologyNode.Config, inputShapes)
	case "projection.linear":
		return inferencer.inferLinear(topologyNode.Config, inputShapes)
	case "math.rmsnorm", "math.layernorm", "math.add",
		"activation.gelu", "activation.silu", "activation.swiglu",
		"positional.rope":
		return inferencer.inferIdentity(inputShapes)
	case "shape.view_as_heads":
		return inferencer.inferViewAsHeads(topologyNode.Config, inputShapes)
	case "shape.merge_heads":
		return inferencer.inferMergeHeads(inputShapes)
	case "attention.sdpa", "attention.gqa":
		return inferencer.inferIdentity(inputShapes)
	case "shape.last_token":
		return inferencer.inferLastToken(inputShapes)
	case "shape.concat":
		return inferencer.inferConcat(topologyNode.Config, inputShapes)
	default:
		return inferencer.inferIdentity(inputShapes)
	}
}

func (inferencer *ShapeInferencer) inferEmbedding(
	config map[string]any,
	inputShapes [][]int64,
) ([][]int64, error) {
	hiddenSize, err := inferencer.configInt64(config, "d_model", "hidden_size")

	if err != nil {
		return nil, err
	}

	if len(inputShapes) == 0 {
		return [][]int64{{ast.DynamicDim, ast.DynamicDim, hiddenSize}}, nil
	}

	output := cloneShape(inputShapes[0])

	switch len(output) {
	case 1:
		output = []int64{ast.DynamicDim, hiddenSize}
	case 2:
		output = append(output, hiddenSize)
	default:
		output[len(output)-1] = hiddenSize
	}

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferLinear(
	config map[string]any,
	inputShapes [][]int64,
) ([][]int64, error) {
	outFeatures, err := inferencer.configInt64(config, "out_features")

	if err != nil {
		return nil, err
	}

	if len(inputShapes) == 0 {
		_, inErr := inferencer.configInt64(config, "in_features")

		if inErr != nil {
			return nil, inErr
		}

		return [][]int64{{ast.DynamicDim, outFeatures}}, nil
	}

	output := cloneShape(inputShapes[0])

	if len(output) == 0 {
		output = []int64{outFeatures}
	}

	output[len(output)-1] = outFeatures

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferViewAsHeads(
	config map[string]any,
	inputShapes [][]int64,
) ([][]int64, error) {
	numHeads, err := inferencer.configInt64(config, "num_heads")

	if err != nil {
		return nil, err
	}

	if len(inputShapes) == 0 || len(inputShapes[0]) == 0 {
		return [][]int64{ast.NewDynamicShape(4)}, nil
	}

	input := inputShapes[0]
	hiddenSize := input[len(input)-1]

	headDim := ast.DynamicDim

	if hiddenSize > 0 {
		if hiddenSize%numHeads != 0 {
			return nil, fmt.Errorf("hidden size %d is not divisible by num_heads %d", hiddenSize, numHeads)
		}

		headDim = hiddenSize / numHeads
	}

	output := append(cloneShape(input[:len(input)-1]), numHeads, headDim)

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferMergeHeads(inputShapes [][]int64) ([][]int64, error) {
	if len(inputShapes) == 0 || len(inputShapes[0]) < 2 {
		return [][]int64{ast.NewDynamicShape(3)}, nil
	}

	input := inputShapes[0]
	numHeads := input[len(input)-2]
	headDim := input[len(input)-1]

	hiddenSize := ast.DynamicDim

	if numHeads > 0 && headDim > 0 {
		hiddenSize = numHeads * headDim
	}

	output := cloneShape(input[:len(input)-2])
	output = append(output, hiddenSize)

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferLastToken(inputShapes [][]int64) ([][]int64, error) {
	if len(inputShapes) == 0 || len(inputShapes[0]) < 2 {
		return [][]int64{ast.NewDynamicShape(2)}, nil
	}

	input := inputShapes[0]
	output := []int64{input[0], input[len(input)-1]}

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferConcat(
	config map[string]any,
	inputShapes [][]int64,
) ([][]int64, error) {
	if len(inputShapes) == 0 {
		return [][]int64{ast.NewDynamicShape(3)}, nil
	}

	axis, err := inferencer.configInt64(config, "dim")

	if err != nil {
		axis = 1
	}

	output := cloneShape(inputShapes[0])

	for _, inputShape := range inputShapes[1:] {
		if len(output) != len(inputShape) {
			return nil, fmt.Errorf("concat rank mismatch")
		}

		if axis < 0 || int(axis) >= len(output) {
			return nil, fmt.Errorf("concat axis %d out of range", axis)
		}

		if output[axis] > 0 && inputShape[axis] > 0 {
			output[axis] += inputShape[axis]
			continue
		}

		output[axis] = ast.DynamicDim
	}

	return [][]int64{output}, nil
}

func (inferencer *ShapeInferencer) inferIdentity(inputShapes [][]int64) ([][]int64, error) {
	if len(inputShapes) == 0 {
		return [][]int64{ast.NewDynamicShape(1)}, nil
	}

	return [][]int64{cloneShape(inputShapes[0])}, nil
}

func (inferencer *ShapeInferencer) configInt64(config map[string]any, keys ...string) (int64, error) {
	for _, key := range keys {
		value, ok := config[key]

		if !ok {
			continue
		}

		return dtype.Int64Value(value)
	}

	return 0, fmt.Errorf("missing config key %q", keys[0])
}

func cloneShape(shape []int64) []int64 {
	if shape == nil {
		return nil
	}

	cloned := make([]int64, len(shape))
	copy(cloned, shape)

	return cloned
}
