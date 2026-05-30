package metal

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/optimizer"
)

/*
SourceKernel is one FusionAST lowered into Metal Shading Language source.
*/
type SourceKernel struct {
	identifier string
	source     string
	inputs     []string
	output     string
	kernelName string
}

func (kernel *SourceKernel) Identifier() string {
	return kernel.identifier
}

func (kernel *SourceKernel) Source() string {
	return kernel.source
}

func (kernel *SourceKernel) KernelName() string {
	return kernel.kernelName
}

func (kernel *SourceKernel) Inputs() []string {
	out := make([]string, len(kernel.inputs))
	copy(out, kernel.inputs)

	return out
}

func (kernel *SourceKernel) Output() string {
	return kernel.output
}

/*
EmitMSL generates MSL source for one FusionAST.
*/
func EmitMSL(fusion *optimizer.FusionAST) (*SourceKernel, error) {
	if fusion == nil {
		return nil, fmt.Errorf("codegen metal: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen metal: fusion root is required")
	}

	kernelName := kernelNameForFusion(fusion)

	var builder strings.Builder

	builder.WriteString("#include <metal_stdlib>\n")
	builder.WriteString("using namespace metal;\n\n")
	builder.WriteString("kernel void ")
	builder.WriteString(kernelName)
	builder.WriteString("(\n")

	for inputIndex := range fusion.InputPorts {
		_, err := fmt.Fprintf(
			&builder,
			"    device const float* in%d [[buffer(%d)]],\n",
			inputIndex, inputIndex,
		)

		if err != nil {
			return nil, fmt.Errorf("codegen metal: write input binding: %w", err)
		}
	}

	if _, err := fmt.Fprintf(
		&builder,
		"    device float* out [[buffer(%d)]],\n",
		len(fusion.InputPorts),
	); err != nil {
		return nil, fmt.Errorf("codegen metal: write output binding: %w", err)
	}

	if _, err := fmt.Fprintf(
		&builder,
		"    constant uint& count [[buffer(%d)]],\n",
		len(fusion.InputPorts)+1,
	); err != nil {
		return nil, fmt.Errorf("codegen metal: write count binding: %w", err)
	}

	builder.WriteString("    uint id [[thread_position_in_grid]]\n")
	builder.WriteString(") {\n")
	builder.WriteString("    if (id >= count) return;\n")

	expression, err := emitMetalExpression(fusion.Root)

	if err != nil {
		return nil, err
	}

	if _, err := fmt.Fprintf(&builder, "    out[id] = %s;\n", expression); err != nil {
		return nil, fmt.Errorf("codegen metal: write body: %w", err)
	}

	builder.WriteString("}\n")

	return &SourceKernel{
		identifier: fusion.OutputPort,
		source:     builder.String(),
		inputs:     append([]string(nil), fusion.InputPorts...),
		output:     fusion.OutputPort,
		kernelName: kernelName,
	}, nil
}

func emitMetalExpression(node *optimizer.ASTNode) (string, error) {
	switch node.Type {
	case optimizer.NodeInput:
		return fmt.Sprintf("in%d[id]", node.InputIndex), nil
	case optimizer.NodeConstant:
		return fmt.Sprintf("(float)%g", node.Value), nil
	case optimizer.NodeAdd:
		return emitMetalBinary("+", node)
	case optimizer.NodeSub:
		return emitMetalBinary("-", node)
	case optimizer.NodeMul:
		return emitMetalBinary("*", node)
	case optimizer.NodeDiv:
		return emitMetalBinary("/", node)
	case optimizer.NodeMax:
		return emitMetalCall2("max", node)
	case optimizer.NodeMin:
		return emitMetalCall2("min", node)
	case optimizer.NodeNeg:
		return emitMetalUnaryPrefix("-", node)
	case optimizer.NodeAbs:
		return emitMetalCall1("fabs", node)
	case optimizer.NodeSqrt:
		return emitMetalCall1("sqrt", node)
	case optimizer.NodeExp:
		return emitMetalCall1("exp", node)
	case optimizer.NodeLog:
		return emitMetalCall1("log", node)
	case optimizer.NodeReLU:
		operand, err := emitMetalExpression(node.Children[0])

		if err != nil {
			return "", err
		}

		return fmt.Sprintf("fmax(0.0f, %s)", operand), nil
	case optimizer.NodeSigmoid:
		operand, err := emitMetalExpression(node.Children[0])

		if err != nil {
			return "", err
		}

		return fmt.Sprintf("(1.0f / (1.0f + exp(-(%s))))", operand), nil
	case optimizer.NodeTanh:
		return emitMetalCall1("tanh", node)
	case optimizer.NodeSilu:
		operand, err := emitMetalExpression(node.Children[0])

		if err != nil {
			return "", err
		}

		return fmt.Sprintf("((%s) / (1.0f + exp(-(%s))))", operand, operand), nil
	case optimizer.NodeGelu:
		operand, err := emitMetalExpression(node.Children[0])

		if err != nil {
			return "", err
		}

		return fmt.Sprintf(
			"(0.5f * (%s) * (1.0f + tanh(0.7978845608028654f * ((%s) + 0.044715f * (%s) * (%s) * (%s)))))",
			operand, operand, operand, operand, operand,
		), nil
	case optimizer.NodeLeakyReLU:
		operand, err := emitMetalExpression(node.Children[0])

		if err != nil {
			return "", err
		}

		return fmt.Sprintf("((%s) >= 0.0f ? (%s) : 0.01f * (%s))", operand, operand, operand), nil
	default:
		return "", fmt.Errorf("codegen metal: unsupported node type %v", node.Type)
	}
}

func emitMetalBinary(operator string, node *optimizer.ASTNode) (string, error) {
	left, err := emitMetalExpression(node.Children[0])

	if err != nil {
		return "", err
	}

	right, err := emitMetalExpression(node.Children[1])

	if err != nil {
		return "", err
	}

	return fmt.Sprintf("((%s) %s (%s))", left, operator, right), nil
}

func emitMetalUnaryPrefix(operator string, node *optimizer.ASTNode) (string, error) {
	operand, err := emitMetalExpression(node.Children[0])

	if err != nil {
		return "", err
	}

	return fmt.Sprintf("(%s(%s))", operator, operand), nil
}

func emitMetalCall1(function string, node *optimizer.ASTNode) (string, error) {
	operand, err := emitMetalExpression(node.Children[0])

	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s(%s)", function, operand), nil
}

func emitMetalCall2(function string, node *optimizer.ASTNode) (string, error) {
	left, err := emitMetalExpression(node.Children[0])

	if err != nil {
		return "", err
	}

	right, err := emitMetalExpression(node.Children[1])

	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s(%s, %s)", function, left, right), nil
}

func kernelNameForFusion(fusion *optimizer.FusionAST) string {
	suffix := strings.ReplaceAll(fusion.OutputPort, ".", "_")
	suffix = strings.ReplaceAll(suffix, "-", "_")

	if suffix == "" {
		suffix = "anon"
	}

	return "fused_" + suffix
}
