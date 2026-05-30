package codegen

/*
MetalFusionProgram is a compiled or compile-ready Metal elementwise fusion
kernel. puter/device/metal/fusion compiles MSL on the active bridge device
and dispatches against resident MTLBuffer handles.
*/
type MetalFusionProgram interface {
	ElementwiseRunner
	MSLSource() string
	MSLKernelName() string
}
