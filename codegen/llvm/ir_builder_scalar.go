package llvm

import (
	"github.com/llir/llvm/ir"
	"github.com/llir/llvm/ir/constant"
	"github.com/llir/llvm/ir/enum"
	"github.com/llir/llvm/ir/types"
	"github.com/llir/llvm/ir/value"
	"github.com/theapemachine/manifesto/optimizer"
)

func (builder *irBuilder) buildScalarLoop(
	entryBlock *ir.Block,
	loopEndBlock *ir.Block,
	startIndex value.Value,
) {
	loopBlock := builder.function.NewBlock("scalar.loop")
	loopBodyBlock := builder.function.NewBlock("scalar.body")

	entryBlock.NewBr(loopBlock)

	loopIndex := loopBlock.NewPhi(ir.NewIncoming(startIndex, entryBlock))
	compare := loopBlock.NewICmp(enum.IPredSGE, loopIndex, builder.countParam)
	loopBlock.NewCondBr(compare, loopEndBlock, loopBodyBlock)

	index64 := loopBodyBlock.NewSExt(loopIndex, int64Type)
	elementValue := builder.emitNode(loopBodyBlock, builder.fusion.Root, index64)
	outputPointer := builder.gepFloat(loopBodyBlock, builder.outputParam, index64)
	loopBodyBlock.NewStore(elementValue, outputPointer)

	nextIndex := loopBodyBlock.NewAdd(loopIndex, constant.NewInt(int32Type, 1))
	loopBodyBlock.NewBr(loopBlock)
	loopIndex.Incs = append(loopIndex.Incs, ir.NewIncoming(nextIndex, loopBodyBlock))
}

func (builder *irBuilder) emitNode(
	block *ir.Block,
	node *optimizer.ASTNode,
	index value.Value,
) value.Value {
	switch node.Type {
	case optimizer.NodeInput:
		inputIndex := constant.NewInt(int64Type, int64(node.InputIndex))
		inputSlot := block.NewGetElementPtr(floatPointer, builder.inputsParam, inputIndex)
		inputBase := block.NewLoad(floatPointer, inputSlot)
		pointer := builder.gepFloat(block, inputBase, index)

		return block.NewLoad(floatType, pointer)
	case optimizer.NodeConstant:
		return constant.NewFloat(floatType, node.Value)
	case optimizer.NodeAdd:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)

		return block.NewFAdd(left, right)
	case optimizer.NodeSub:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)

		return block.NewFSub(left, right)
	case optimizer.NodeMul:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)

		return block.NewFMul(left, right)
	case optimizer.NodeDiv:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)

		return block.NewFDiv(left, right)
	case optimizer.NodeMax:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)
		compare := block.NewFCmp(enum.FPredOGT, left, right)

		return block.NewSelect(compare, left, right)
	case optimizer.NodeMin:
		left := builder.emitNode(block, node.Children[0], index)
		right := builder.emitNode(block, node.Children[1], index)
		compare := block.NewFCmp(enum.FPredOLT, left, right)

		return block.NewSelect(compare, left, right)
	case optimizer.NodeNeg:
		valueInput := builder.emitNode(block, node.Children[0], index)

		return block.NewFNeg(valueInput)
	case optimizer.NodeAbs:
		valueInput := builder.emitNode(block, node.Children[0], index)
		zero := constant.NewFloat(floatType, 0)
		compare := block.NewFCmp(enum.FPredOLT, valueInput, zero)
		negated := block.NewFNeg(valueInput)

		return block.NewSelect(compare, negated, valueInput)
	case optimizer.NodeSqrt:
		return builder.callMath(block, "sqrtf", builder.emitNode(block, node.Children[0], index))
	case optimizer.NodeExp:
		return builder.callMath(block, "expf", builder.emitNode(block, node.Children[0], index))
	case optimizer.NodeLog:
		return builder.callMath(block, "logf", builder.emitNode(block, node.Children[0], index))
	case optimizer.NodeReLU:
		valueInput := builder.emitNode(block, node.Children[0], index)
		zero := constant.NewFloat(floatType, 0)
		compare := block.NewFCmp(enum.FPredOLT, valueInput, zero)

		return block.NewSelect(compare, zero, valueInput)
	case optimizer.NodeSigmoid:
		valueInput := builder.emitNode(block, node.Children[0], index)
		negated := block.NewFNeg(valueInput)
		expValue := builder.callMath(block, "expf", negated)
		one := constant.NewFloat(floatType, 1)

		return block.NewFDiv(one, block.NewFAdd(one, expValue))
	case optimizer.NodeTanh:
		return builder.callMath(block, "tanhf", builder.emitNode(block, node.Children[0], index))
	case optimizer.NodeSilu:
		valueInput := builder.emitNode(block, node.Children[0], index)
		negated := block.NewFNeg(valueInput)
		expValue := builder.callMath(block, "expf", negated)
		one := constant.NewFloat(floatType, 1)

		return block.NewFDiv(valueInput, block.NewFAdd(one, expValue))
	case optimizer.NodeGelu:
		return builder.emitGelu(block, node.Children[0], index)
	case optimizer.NodeLeakyReLU:
		valueInput := builder.emitNode(block, node.Children[0], index)
		zero := constant.NewFloat(floatType, 0)
		compare := block.NewFCmp(enum.FPredOGE, valueInput, zero)
		scaled := block.NewFMul(valueInput, constant.NewFloat(floatType, 0.01))

		return block.NewSelect(compare, valueInput, scaled)
	default:
		return constant.NewFloat(floatType, 0)
	}
}

func (builder *irBuilder) emitGelu(
	block *ir.Block,
	child *optimizer.ASTNode,
	index value.Value,
) value.Value {
	valueInput := builder.emitNode(block, child, index)
	value64 := block.NewFPExt(valueInput, types.Double)

	cube := block.NewFMul(value64, block.NewFMul(value64, value64))
	coeff := constant.NewFloat(types.Double, 0.044715)
	cubicTerm := block.NewFMul(coeff, cube)
	innerSum := block.NewFAdd(value64, cubicTerm)
	sqrtTwoPi := constant.NewFloat(types.Double, 0.7978845608028654)
	inner := block.NewFMul(sqrtTwoPi, innerSum)
	inner32 := block.NewFPTrunc(inner, floatType)
	tanhValue := builder.callMath(block, "tanhf", inner32)
	tanh64 := block.NewFPExt(tanhValue, types.Double)
	one := constant.NewFloat(types.Double, 1)
	onePlusTanh := block.NewFAdd(one, tanh64)
	half := constant.NewFloat(types.Double, 0.5)
	scale := block.NewFMul(half, onePlusTanh)
	result := block.NewFMul(value64, scale)

	return block.NewFPTrunc(result, floatType)
}

func (builder *irBuilder) gepFloat(
	block *ir.Block,
	base value.Value,
	index value.Value,
) value.Value {
	return block.NewGetElementPtr(floatType, base, index)
}

func (builder *irBuilder) declareMathFunctions() {
	for _, name := range []string{"sqrtf", "expf", "logf", "tanhf"} {
		builder.mathDecls[name] = builder.module.NewFunc(name, floatType, ir.NewParam("x", floatType))
	}
}

func (builder *irBuilder) callMath(
	block *ir.Block,
	name string,
	argument value.Value,
) value.Value {
	return block.NewCall(builder.mathDecls[name], argument)
}
