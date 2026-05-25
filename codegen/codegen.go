/*
Package codegen lowers manifesto/optimizer FusionAST clusters into target
backends. Per ARCHITECTURE.md §4.3 and the "Detailed Implementation
Blueprints" §2, the codegen layer is responsible for emitting one kernel
per fused subgraph; downstream targets (CPU LLVM JIT, Metal MSL, CUDA
PTX, XLA HLO) all consume the same FusionAST.

This package owns the lowering itself, not execution. Kernels returned by
codegen are stored on the optimizer-emitted fused GraphNode under
KernelAttribute and invoked by puter/execution at dispatch time.

The package targets CPU (in-process Go evaluator) and Metal (MSL source
generator) per the user's "CPU + Metal only" current direction. CUDA and
XLA paths fit cleanly behind the same Kernel contract once they're needed.
*/
package codegen

import (
	"github.com/theapemachine/manifesto/optimizer"
)

/*
KernelAttribute is the ast.GraphNode.Attributes key under which a compiled
Kernel is stored. The execution backend looks for this key on every
FuseOp node and invokes the kernel directly — no further dispatch by
op kind is needed.
*/
const KernelAttribute = "compiled_kernel"

/*
Target identifies which backend a Kernel was emitted for. The execution
backend picks the matching kernel for the active device.
*/
type Target int

const (
	// TargetCPU is the in-process Go evaluator (no LLVM yet).
	TargetCPU Target = iota
	// TargetMetal is MSL source intended for compilation by
	// puter/device/metal at session init.
	TargetMetal
)

/*
Kernel is one compiled FusionAST. Concrete implementations attach extra
fields per target (Go function closure for CPU, MSL source string for
Metal). The executor only needs Target() to decide whether the active
device backend can run the kernel and Identifier() for diagnostics.
*/
type Kernel interface {
	Target() Target
	Identifier() string
}

/*
EmitOptions configures codegen. Targets is the set of backends the caller
wants kernels for; an empty set defaults to {TargetCPU, TargetMetal}.
*/
type EmitOptions struct {
	Targets []Target
}

/*
EmitFusion lowers one FusionAST into the requested kernels. The returned
slice is parallel to the requested target list (or to the default list
when none was given).
*/
func EmitFusion(fusion *optimizer.FusionAST, options EmitOptions) ([]Kernel, error) {
	targets := options.Targets

	if len(targets) == 0 {
		targets = []Target{TargetCPU, TargetMetal}
	}

	kernels := make([]Kernel, 0, len(targets))

	for _, target := range targets {
		switch target {
		case TargetCPU:
			kernel, err := EmitCPU(fusion)

			if err != nil {
				return nil, err
			}

			kernels = append(kernels, kernel)
		case TargetMetal:
			kernel, err := EmitMetal(fusion)

			if err != nil {
				return nil, err
			}

			kernels = append(kernels, kernel)
		}
	}

	return kernels, nil
}
