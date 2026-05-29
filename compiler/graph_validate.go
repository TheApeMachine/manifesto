package compiler

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/typer"
	"github.com/theapemachine/manifesto/types"
)

/*
validateGraphOps enforces the closed-world operation contract from
ARCHITECTURE.md §2.1: every node must resolve to a registered manifest
operation with bind.method, a compiler intrinsic handled outside the
static kernel table, or a typer adaptor the synthesizer emitted.
*/
func validateGraphOps(graph *ast.Graph, registry *types.OperationRegistry) error {
	if graph == nil {
		return fmt.Errorf("compiler: graph is required")
	}

	if registry == nil {
		return fmt.Errorf("compiler: operation registry is required")
	}

	for _, node := range graph.Nodes {
		if node == nil {
			return fmt.Errorf("compiler: graph contains a nil node")
		}

		if err := validateOneGraphOp(node, registry); err != nil {
			return err
		}
	}

	return nil
}

func validateOneGraphOp(node *ast.GraphNode, registry *types.OperationRegistry) error {
	op := strings.TrimSpace(node.Op)

	if op == "" {
		return fmt.Errorf("compiler: node %q: op is required", node.ID)
	}

	if op == optimizer.FuseOp {
		return validateFusedNode(node)
	}

	if isCompilerIntrinsicOp(op) {
		return nil
	}

	schema, inRegistry := registry.Lookup(types.Op(op))

	if inRegistry {
		if schema.Bind == nil || strings.TrimSpace(schema.Bind.Method) == "" {
			return fmt.Errorf("compiler: node %q: operation %q has no bind.method", node.ID, op)
		}

		return nil
	}

	if _, inTyper := typer.LookupSpec(op); inTyper {
		return nil
	}

	return fmt.Errorf("compiler: node %q: operation %q is not registered", node.ID, op)
}

func validateFusedNode(node *ast.GraphNode) error {
	fusionAny, ok := node.Attributes[optimizer.FuseAttributeAST]

	if !ok {
		return fmt.Errorf(
			"compiler: fused node %q missing %q attribute",
			node.ID, optimizer.FuseAttributeAST,
		)
	}

	if _, ok := fusionAny.(*optimizer.FusionAST); !ok {
		return fmt.Errorf(
			"compiler: fused node %q has invalid %q attribute type %T",
			node.ID, optimizer.FuseAttributeAST, fusionAny,
		)
	}

	return nil
}

func isCompilerIntrinsicOp(op string) bool {
	switch op {
	case "value.assign":
		return true
	default:
		return false
	}
}
