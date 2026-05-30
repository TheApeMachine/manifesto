//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "fusion_jit.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

type jitRunner struct {
	handle C.MetalFusionJITRef
}

func (runner *jitRunner) close() {
	if runner == nil || runner.handle == nil {
		return
	}

	C.metal_fusion_jit_release(runner.handle)
	runner.handle = nil
}

func (runner *jitRunner) run(
	inputs [][]float32,
	output []float32,
	count int,
) error {
	if count == 0 {
		return nil
	}

	if runner == nil || runner.handle == nil {
		return fmt.Errorf("codegen metal: jit runner is closed")
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()

	for inputIndex := range inputs {
		pinner.Pin(&inputs[inputIndex][0])
	}

	pinner.Pin(&output[0])

	inputSlots := C.calloc(C.size_t(len(inputs)), C.size_t(unsafe.Sizeof(uintptr(0))))

	if inputSlots == nil {
		return fmt.Errorf("codegen metal: allocate input pointer table")
	}

	defer C.free(inputSlots)

	slotSlice := unsafe.Slice((**C.float)(inputSlots), len(inputs))

	for inputIndex, buffer := range inputs {
		slotSlice[inputIndex] = (*C.float)(unsafe.Pointer(&buffer[0]))
	}

	status := C.MetalFusionStatus{}
	code := C.metal_fusion_jit_run_host(
		runner.handle,
		(**C.float)(inputSlots),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(inputs)),
		C.uint32_t(count),
		&status,
	)

	if code != 0 {
		return fmt.Errorf("codegen metal: dispatch: %s", C.GoString(&status.message[0]))
	}

	return nil
}

func compileMSL(source, kernelName string) (fusionRunner, error) {
	status := C.MetalFusionStatus{}
	sourceCString := C.CString(source)
	kernelCString := C.CString(kernelName)

	defer C.free(unsafe.Pointer(sourceCString))
	defer C.free(unsafe.Pointer(kernelCString))

	handle := C.metal_fusion_jit_compile(sourceCString, kernelCString, &status)

	if handle == nil {
		return nil, fmt.Errorf("codegen metal: compile %q: %s", kernelName, C.GoString(&status.message[0]))
	}

	return &jitRunner{handle: handle}, nil
}
