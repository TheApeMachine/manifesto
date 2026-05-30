//go:build codegen_llvm

package llvm

import (
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/manifesto/optimizer"
)

var (
	jitInitOnce sync.Once
	jitInitErr  error
)

/*
EmitKernel lowers fusion into LLVM IR, JIT-compiles it for the host CPU
with CPUID-selected target features, and returns an executable kernel.
*/
func EmitKernel(fusion *optimizer.FusionAST) (*CompiledKernel, error) {
	if err := ValidateHostJITSupport(); err != nil {
		return nil, err
	}

	jitInitOnce.Do(func() {
		jitInitErr = initializeLLVMRuntime()
	})

	if jitInitErr != nil {
		return nil, jitInitErr
	}

	if fusion == nil {
		return nil, fmt.Errorf("codegen llvm: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen llvm: fusion root is required")
	}

	identifier := fusion.OutputPort

	if identifier == "" {
		identifier = "anon_" + strings.Join(fusion.InputPorts, "_")
	}

	functionName := sanitizeFunctionName(identifier)
	irText, err := ModuleIR(fusion, functionName)

	if err != nil {
		return nil, err
	}

	runner, err := compileIRFunction(irText, functionName, len(fusion.InputPorts))

	if err != nil {
		return nil, fmt.Errorf("codegen llvm: jit %q: %w", functionName, err)
	}

	return newCompiledKernel(fusion, runner), nil
}

/*
EmitKernelForLevel JIT-compiles fusion with LLVM target features for one ISA
tier. Used by parity tests that exercise each SIMD level the host supports.
*/
func EmitKernelForLevel(fusion *optimizer.FusionAST, level ISALevel) (*CompiledKernel, error) {
	if err := ValidateHostJITSupport(); err != nil {
		return nil, err
	}

	jitInitOnce.Do(func() {
		jitInitErr = initializeLLVMRuntime()
	})

	if jitInitErr != nil {
		return nil, jitInitErr
	}

	if fusion == nil {
		return nil, fmt.Errorf("codegen llvm: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen llvm: fusion root is required")
	}

	identifier := fusion.OutputPort

	if identifier == "" {
		identifier = "anon_" + strings.Join(fusion.InputPorts, "_")
	}

	functionName := sanitizeFunctionName(identifier)
	irText, err := ModuleIR(fusion, functionName)

	if err != nil {
		return nil, err
	}

	runner, err := compileIRFunctionForLevel(irText, functionName, len(fusion.InputPorts), level)

	if err != nil {
		return nil, fmt.Errorf("codegen llvm: jit %q (%s): %w", functionName, level, err)
	}

	return newCompiledKernel(fusion, runner), nil
}
