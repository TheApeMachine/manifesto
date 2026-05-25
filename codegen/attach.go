package codegen

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/optimizer"
)

/*
KernelSet groups every emitted kernel for one FusionAST, keyed by target.
This is what gets stored under KernelAttribute on the FuseOp node.
*/
type KernelSet struct {
	kernels map[Target]Kernel
}

/*
NewKernelSet constructs a KernelSet from a slice of Kernels (typically the
result of EmitFusion).
*/
func NewKernelSet(kernels []Kernel) *KernelSet {
	set := &KernelSet{kernels: make(map[Target]Kernel, len(kernels))}

	for _, kernel := range kernels {
		set.kernels[kernel.Target()] = kernel
	}

	return set
}

/*
For returns the kernel emitted for the given target, or nil if none was
emitted.
*/
func (set *KernelSet) For(target Target) Kernel {
	if set == nil {
		return nil
	}

	return set.kernels[target]
}

/*
Targets returns the set of targets that have kernels in this KernelSet.
*/
func (set *KernelSet) Targets() []Target {
	if set == nil {
		return nil
	}

	out := make([]Target, 0, len(set.kernels))

	for target := range set.kernels {
		out = append(out, target)
	}

	return out
}

/*
AttachKernels walks every FuseOp node on graph, emits kernels per
EmitOptions, and stores the resulting KernelSet under KernelAttribute.
Non-FuseOp nodes are left untouched.

Returns the number of fused nodes for which kernels were attached.
*/
func AttachKernels(graph *ast.Graph, options EmitOptions) (int, error) {
	if graph == nil {
		return 0, fmt.Errorf("codegen: graph is required")
	}

	attached := 0

	for _, node := range graph.Nodes {
		if node == nil || node.Op != optimizer.FuseOp {
			continue
		}

		fusionAny, ok := node.Attributes[optimizer.FuseAttributeAST]

		if !ok {
			return attached, fmt.Errorf(
				"codegen: fused node %q missing %q attribute",
				node.ID, optimizer.FuseAttributeAST,
			)
		}

		fusion, ok := fusionAny.(*optimizer.FusionAST)

		if !ok {
			return attached, fmt.Errorf(
				"codegen: fused node %q has %q of type %T, want *optimizer.FusionAST",
				node.ID, optimizer.FuseAttributeAST, fusionAny,
			)
		}

		kernels, err := EmitFusion(fusion, options)

		if err != nil {
			return attached, fmt.Errorf("codegen: emit %q: %w", node.ID, err)
		}

		if node.Attributes == nil {
			node.Attributes = make(map[string]any)
		}

		node.Attributes[KernelAttribute] = NewKernelSet(kernels)
		attached++
	}

	return attached, nil
}
