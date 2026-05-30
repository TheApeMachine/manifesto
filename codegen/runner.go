package codegen

/*
ElementwiseRunner is the execution contract for CPU elementwise fusion
kernels. Both the scalar reference evaluator and the LLVM JIT kernel
implement this interface so puter/execution can dispatch without caring
which codegen backend produced the kernel.
*/
type ElementwiseRunner interface {
	Kernel
	Inputs() []string
	Output() string
	Run(inputs [][]float32, output []float32, count int) error
}
