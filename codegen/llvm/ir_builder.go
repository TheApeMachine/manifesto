package llvm

import (
	"fmt"
	"strings"

	"github.com/llir/llvm/ir"
	"github.com/llir/llvm/ir/constant"
	"github.com/llir/llvm/ir/types"
	"github.com/theapemachine/manifesto/optimizer"
)

var (
	floatType       = types.Float
	floatPointer    = types.NewPointer(types.Float)
	floatPointerPtr = types.NewPointer(floatPointer)
	int32Type       = types.I32
	int64Type       = types.I64
)

/*
BuildModule lowers one FusionAST into an LLVM IR module containing a
single void kernel:

	void fused_<name>(float** inputs, float* out, i32 count)
*/
func BuildModule(fusion *optimizer.FusionAST, functionName string) (*ir.Module, error) {
	if fusion == nil {
		return nil, fmt.Errorf("codegen llvm: fusion is required")
	}

	if fusion.Root == nil {
		return nil, fmt.Errorf("codegen llvm: fusion root is required")
	}

	if functionName == "" {
		return nil, fmt.Errorf("codegen llvm: function name is required")
	}

	if err := validateFusionAST(fusion); err != nil {
		return nil, err
	}

	builder := newIRBuilder(ir.NewModule(), fusion, sanitizeFunctionName(functionName))

	if err := builder.build(); err != nil {
		return nil, err
	}

	return builder.module, nil
}

/*
ModuleIR returns the textual LLVM IR for one FusionAST.
*/
func ModuleIR(fusion *optimizer.FusionAST, functionName string) (string, error) {
	module, err := BuildModule(fusion, functionName)

	if err != nil {
		return "", err
	}

	return module.String(), nil
}

type irBuilder struct {
	module       *ir.Module
	fusion       *optimizer.FusionAST
	functionName string
	function     *ir.Func
	inputsParam  *ir.Param
	outputParam  *ir.Param
	countParam   *ir.Param
	mathDecls    map[string]*ir.Func
	vectorWidth  int
	vectorType   *types.VectorType
}

func newIRBuilder(
	module *ir.Module,
	fusion *optimizer.FusionAST,
	functionName string,
) *irBuilder {
	vectorWidth := HostVectorWidth()

	return &irBuilder{
		module:       module,
		fusion:       fusion,
		functionName: functionName,
		mathDecls:    make(map[string]*ir.Func),
		vectorWidth:  vectorWidth,
		vectorType:   types.NewVector(uint64(vectorWidth), floatType),
	}
}

func (builder *irBuilder) build() error {
	builder.declareMathFunctions()

	builder.function = builder.module.NewFunc(
		builder.functionName,
		types.Void,
		ir.NewParam("inputs", floatPointerPtr),
		ir.NewParam("out", floatPointer),
		ir.NewParam("count", int32Type),
	)
	builder.inputsParam = builder.function.Params[0]
	builder.outputParam = builder.function.Params[1]
	builder.countParam = builder.function.Params[2]

	entryBlock := builder.function.NewBlock("entry")
	loopEndBlock := builder.function.NewBlock("loop.end")

	if fusionVectorizable(builder.fusion.Root) && builder.vectorWidth > 1 {
		builder.buildVectorThenScalar(entryBlock, loopEndBlock)
	} else {
		builder.buildScalarLoop(entryBlock, loopEndBlock, constant.NewInt(int32Type, 0))
	}

	loopEndBlock.NewRet(nil)

	return nil
}

func sanitizeFunctionName(name string) string {
	replaced := strings.NewReplacer(
		".", "_",
		"-", "_",
		" ", "_",
	).Replace(name)

	if replaced == "" {
		return "fused_anon"
	}

	if strings.HasPrefix(replaced, "fused_") {
		return replaced
	}

	return "fused_" + replaced
}
