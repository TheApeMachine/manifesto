package llvm

import (
	"github.com/llir/llvm/ir"
	"github.com/llir/llvm/ir/constant"
	"github.com/llir/llvm/ir/enum"
	"github.com/llir/llvm/ir/types"
	"github.com/llir/llvm/ir/value"
	"github.com/theapemachine/manifesto/optimizer"
)

func (builder *irBuilder) buildVectorThenScalar(
	entryBlock *ir.Block,
	loopEndBlock *ir.Block,
) {
	vectorWidthConst := constant.NewInt(int32Type, int64(builder.vectorWidth))
	vectorRemainder := entryBlock.NewSRem(builder.countParam, vectorWidthConst)
	vectorLimit := entryBlock.NewSub(builder.countParam, vectorRemainder)

	vectorLoopBlock := builder.function.NewBlock("vector.loop")
	vectorBodyBlock := builder.function.NewBlock("vector.body")
	scalarEntryBlock := builder.function.NewBlock("scalar.entry")

	entryBlock.NewBr(vectorLoopBlock)

	zero := constant.NewInt(int32Type, 0)
	vectorIndex := vectorLoopBlock.NewPhi(ir.NewIncoming(zero, entryBlock))
	vectorCompare := vectorLoopBlock.NewICmp(enum.IPredSGE, vectorIndex, vectorLimit)
	vectorLoopBlock.NewCondBr(vectorCompare, scalarEntryBlock, vectorBodyBlock)

	index64 := vectorBodyBlock.NewSExt(vectorIndex, int64Type)
	vectorValue := builder.emitVectorNode(vectorBodyBlock, builder.fusion.Root, index64)
	builder.storeVector(vectorBodyBlock, builder.outputParam, index64, vectorValue)

	nextVectorIndex := vectorBodyBlock.NewAdd(vectorIndex, vectorWidthConst)
	vectorBodyBlock.NewBr(vectorLoopBlock)
	vectorIndex.Incs = append(vectorIndex.Incs, ir.NewIncoming(nextVectorIndex, vectorBodyBlock))

	builder.buildScalarLoop(scalarEntryBlock, loopEndBlock, vectorLimit)
}

func (builder *irBuilder) emitVectorNode(
	block *ir.Block,
	node *optimizer.ASTNode,
	index value.Value,
) value.Value {
	switch node.Type {
	case optimizer.NodeInput:
		inputIndex := constant.NewInt(int64Type, int64(node.InputIndex))
		inputSlot := block.NewGetElementPtr(floatPointer, builder.inputsParam, inputIndex)
		inputBase := block.NewLoad(floatPointer, inputSlot)

		return builder.loadVector(block, inputBase, index)
	case optimizer.NodeConstant:
		return builder.vectorSplat(block, node.Value)
	case optimizer.NodeAdd:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)

		return block.NewFAdd(left, right)
	case optimizer.NodeSub:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)

		return block.NewFSub(left, right)
	case optimizer.NodeMul:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)

		return block.NewFMul(left, right)
	case optimizer.NodeDiv:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)

		return block.NewFDiv(left, right)
	case optimizer.NodeMax:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)
		compare := block.NewFCmp(enum.FPredOGT, left, right)

		return block.NewSelect(compare, left, right)
	case optimizer.NodeMin:
		left := builder.emitVectorNode(block, node.Children[0], index)
		right := builder.emitVectorNode(block, node.Children[1], index)
		compare := block.NewFCmp(enum.FPredOLT, left, right)

		return block.NewSelect(compare, left, right)
	case optimizer.NodeNeg:
		valueInput := builder.emitVectorNode(block, node.Children[0], index)

		return block.NewFNeg(valueInput)
	case optimizer.NodeAbs:
		valueInput := builder.emitVectorNode(block, node.Children[0], index)
		zero := builder.vectorSplat(block, 0)
		compare := block.NewFCmp(enum.FPredOLT, valueInput, zero)
		negated := block.NewFNeg(valueInput)

		return block.NewSelect(compare, negated, valueInput)
	case optimizer.NodeReLU:
		valueInput := builder.emitVectorNode(block, node.Children[0], index)
		zero := builder.vectorSplat(block, 0)
		compare := block.NewFCmp(enum.FPredOLT, valueInput, zero)

		return block.NewSelect(compare, zero, valueInput)
	case optimizer.NodeLeakyReLU:
		valueInput := builder.emitVectorNode(block, node.Children[0], index)
		zero := builder.vectorSplat(block, 0)
		compare := block.NewFCmp(enum.FPredOGE, valueInput, zero)
		scaled := block.NewFMul(valueInput, builder.vectorSplat(block, 0.01))

		return block.NewSelect(compare, valueInput, scaled)
	default:
		return builder.vectorSplat(block, 0)
	}
}

func (builder *irBuilder) loadVector(
	block *ir.Block,
	base value.Value,
	index value.Value,
) value.Value {
	scalarPointer := builder.gepFloat(block, base, index)
	vectorPointerType := types.NewPointer(builder.vectorType)
	vectorPointer := block.NewBitCast(scalarPointer, vectorPointerType)

	return block.NewLoad(builder.vectorType, vectorPointer)
}

func (builder *irBuilder) storeVector(
	block *ir.Block,
	base value.Value,
	index value.Value,
	vectorValue value.Value,
) {
	scalarPointer := builder.gepFloat(block, base, index)
	vectorPointerType := types.NewPointer(builder.vectorType)
	vectorPointer := block.NewBitCast(scalarPointer, vectorPointerType)

	block.NewStore(vectorValue, vectorPointer)
}

func (builder *irBuilder) vectorSplat(
	block *ir.Block,
	scalar float64,
) value.Value {
	elements := make([]constant.Constant, builder.vectorWidth)

	for index := 0; index < builder.vectorWidth; index++ {
		elements[index] = constant.NewFloat(floatType, scalar)
	}

	return constant.NewVector(builder.vectorType, elements...)
}
