//go:build codegen_llvm

package llvm

/*
#include <stdint.h>
#include <stdlib.h>

typedef void (*manifesto_kernel_fn)(float **inputs, float *out, int32_t count);

static void manifesto_run_kernel(uint64_t address, float **inputs, float *out, int32_t count) {
	manifesto_kernel_fn kernel = (manifesto_kernel_fn)(uintptr_t)address;
	kernel(inputs, out, count);
}
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	gllvm "tinygo.org/x/go-llvm"
)

type mcjitRunner struct {
	engine     gllvm.ExecutionEngine
	context    gllvm.Context
	address    uint64
	inputCount int
	closed     bool
}

func (runner *mcjitRunner) close() {
	if runner == nil || runner.closed {
		return
	}

	runner.engine.Dispose()
	runner.context.Dispose()
	runner.closed = true
}

func (runner *mcjitRunner) run(
	inputs [][]float32,
	output []float32,
	count int,
) error {
	if count == 0 {
		return nil
	}

	if len(inputs) != runner.inputCount {
		return fmt.Errorf(
			"codegen llvm: jit kernel expects %d inputs, got %d",
			runner.inputCount, len(inputs),
		)
	}

	elementCount := int32(count)

	var pinner runtime.Pinner
	defer pinner.Unpin()

	for inputIndex := range inputs {
		pinner.Pin(&inputs[inputIndex][0])
	}

	pinner.Pin(&output[0])

	inputSlots := C.calloc(C.size_t(len(inputs)), C.size_t(unsafe.Sizeof(uintptr(0))))

	if inputSlots == nil {
		return fmt.Errorf("codegen llvm: allocate input pointer table")
	}

	defer C.free(inputSlots)

	slotSlice := unsafe.Slice((**C.float)(inputSlots), len(inputs))

	for inputIndex, buffer := range inputs {
		slotSlice[inputIndex] = (*C.float)(unsafe.Pointer(&buffer[0]))
	}

	C.manifesto_run_kernel(
		C.uint64_t(runner.address),
		(**C.float)(inputSlots),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int32_t(elementCount),
	)

	return nil
}

func initializeLLVMRuntime() error {
	gllvm.LinkInMCJIT()

	if err := gllvm.InitializeNativeTarget(); err != nil {
		return fmt.Errorf("codegen llvm: initialize native target: %w", err)
	}

	if err := gllvm.InitializeNativeAsmPrinter(); err != nil {
		return fmt.Errorf("codegen llvm: initialize native asm printer: %w", err)
	}

	return nil
}

func parseIRModule(context gllvm.Context, irText string) (gllvm.Module, error) {
	tempFile, err := os.CreateTemp("", "manifesto-fusion-*.ll")

	if err != nil {
		return gllvm.Module{}, fmt.Errorf("codegen llvm: temp ir file: %w", err)
	}

	tempPath := tempFile.Name()

	defer os.Remove(tempPath)

	if _, err := tempFile.WriteString(irText); err != nil {
		tempFile.Close()
		return gllvm.Module{}, fmt.Errorf("codegen llvm: write ir file: %w", err)
	}

	if err := tempFile.Close(); err != nil {
		return gllvm.Module{}, fmt.Errorf("codegen llvm: close ir file: %w", err)
	}

	buffer, err := gllvm.NewMemoryBufferFromFile(tempPath)

	if err != nil {
		return gllvm.Module{}, fmt.Errorf("codegen llvm: read ir file: %w", err)
	}

	module, err := context.ParseIR(buffer)

	if err != nil {
		buffer.Dispose()
		return gllvm.Module{}, err
	}

	if err := gllvm.VerifyModule(module, gllvm.ReturnStatusAction); err != nil {
		buffer.Dispose()
		module.Dispose()
		return gllvm.Module{}, fmt.Errorf("codegen llvm: verify module: %w", err)
	}

	return module, nil
}

func compileIRFunction(irText, functionName string, inputCount int) (kernelRunner, error) {
	return compileIRFunctionForLevel(irText, functionName, inputCount, HostISALevel())
}

func compileIRFunctionForLevel(
	irText string,
	functionName string,
	inputCount int,
	level ISALevel,
) (kernelRunner, error) {
	context := gllvm.NewContext()
	module, err := parseIRModule(context, irText)

	if err != nil {
		context.Dispose()
		return nil, err
	}

	module.SetTarget(HostTargetTriple())

	targetMachine := hostTargetMachineForLevel(level)
	defer targetMachine.Dispose()

	if err := module.RunPasses("default<O3>", targetMachine, gllvm.NewPassBuilderOptions()); err != nil {
		module.Dispose()
		context.Dispose()
		return nil, fmt.Errorf("codegen llvm: optimize module: %w", err)
	}

	options := gllvm.NewMCJITCompilerOptions()
	options.SetMCJITOptimizationLevel(3)
	options.SetMCJITCodeModel(gllvm.CodeModelJITDefault)

	engine, err := gllvm.NewMCJITCompiler(module, options)

	if err != nil {
		context.Dispose()
		module.Dispose()
		return nil, err
	}

	engine.RunStaticConstructors()

	address := engine.GetFunctionAddress(functionName)

	if address == 0 {
		engine.Dispose()
		context.Dispose()
		return nil, fmt.Errorf("codegen llvm: missing function %q", functionName)
	}

	return &mcjitRunner{
		engine:     engine,
		context:    context,
		address:    address,
		inputCount: inputCount,
	}, nil
}

func hostTargetMachine() gllvm.TargetMachine {
	return hostTargetMachineForLevel(HostISALevel())
}

func hostTargetMachineForLevel(level ISALevel) gllvm.TargetMachine {
	triple := HostTargetTriple()
	target, err := gllvm.GetTargetFromTriple(triple)

	if err != nil {
		fmt.Fprintf(os.Stderr, "codegen llvm: target lookup %q: %v\n", triple, err)
		return gllvm.TargetMachine{}
	}

	return target.CreateTargetMachine(
		triple,
		"generic",
		CPUFeaturesForLevel(level),
		gllvm.CodeGenLevelDefault,
		gllvm.RelocDefault,
		gllvm.CodeModelDefault,
	)
}
