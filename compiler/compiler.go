package compiler

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/errnie"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/types"
)

/*
Compiler builds manifest IR from a topology recipe and a parser's tokens.
*/
type Compiler struct {
	ctx      context.Context
	cancel   context.CancelFunc
	pool     *Pool
	parser   types.Parser
	topology *ast.Topology
	registry *types.OperationRegistry
	project  *ir.Project
}

/*
NewCompiler constructs a Compiler over one archive parser.
*/
func NewCompiler(ctx context.Context, pool *Pool, parser types.Parser) (*Compiler, error) {
	ctx, cancel := context.WithCancel(ctx)

	registry, err := types.NewOperationRegistry()

	if err != nil {
		cancel()
		return nil, fmt.Errorf("compiler: %w", err)
	}

	compiler := &Compiler{
		ctx:      ctx,
		cancel:   cancel,
		pool:     pool,
		parser:   parser,
		registry: registry,
	}

	return compiler, errnie.Require(map[string]any{
		"ctx":      compiler.ctx,
		"cancel":   compiler.cancel,
		"parser":   compiler.parser,
		"registry": compiler.registry,
	})
}

/*
WithTopology attaches the expanded or raw topology recipe the compiler
materializes into ir.Topology nodes.
*/
func (compiler *Compiler) WithTopology(topology *ast.Topology) *Compiler {
	if compiler == nil {
		return nil
	}

	compiler.topology = topology

	return compiler
}

/*
CompileTopology lowers the attached topology recipe, binds checkpoint
weights when a parser is configured, and runs the shared graph pipeline.
*/
func (compiler *Compiler) CompileTopology() (*CompiledGraph, error) {
	if compiler.topology == nil {
		return nil, fmt.Errorf("compiler: topology is required")
	}

	lowered, err := LowerTopology(compiler.topology)

	if err != nil {
		return nil, fmt.Errorf("compiler: lower topology: %w", err)
	}

	return CompileGraph(lowered.AST, GraphCompileOptions{
		OperationRegistry: compiler.registry,
		WeightParser:      compiler.parser,
	})
}

/*
Build materializes Project → Architecture → Topology → Node from the
attached topology recipe and checkpoint tokens yielded by the parser.
*/
func (compiler *Compiler) Build() (*ir.Project, error) {
	if compiler.topology == nil {
		return nil, fmt.Errorf("compiler: topology is required")
	}

	compiler.project = &ir.Project{
		Kind:     ir.KindResearchProject,
		Metadata: make(map[string]string),
		Architecture: &ir.Architecture{
			Kind: ir.KindArchitecture,
			Topology: &ir.Topology{
				Kind: ir.KindTopology,
			},
		},
	}

	tokenIndex, err := compiler.indexTokens()

	if err != nil {
		return nil, err
	}

	expanded, err := expandTopology(compiler.topology)

	if err != nil {
		return nil, fmt.Errorf("compiler: expand topology: %w", err)
	}

	for _, topologyNode := range expanded.Nodes {
		node, err := compiler.buildNode(topologyNode, tokenIndex)

		if err != nil {
			return nil, err
		}

		compiler.project.Architecture.Topology.Nodes = append(
			compiler.project.Architecture.Topology.Nodes,
			node,
		)
	}

	return compiler.project, nil
}

func (compiler *Compiler) indexTokens() (map[string]types.Token, error) {
	tokenIndex := make(map[string]types.Token)

	for token := range compiler.parser.Generate() {
		switch token.Kind {
		case types.KindMetadata:
			compiler.project.Metadata[token.Name] = token.Value
		case types.KindTensor:
			tokenIndex[token.Name] = token
		default:
			return nil, fmt.Errorf("compiler: unknown token kind: %d", token.Kind)
		}
	}

	return tokenIndex, nil
}

func (compiler *Compiler) buildNode(
	topologyNode ast.Node,
	tokenIndex map[string]types.Token,
) (*ir.Node, error) {
	op := types.Op(strings.TrimSpace(topologyNode.Op))

	if op == "" {
		return nil, fmt.Errorf("compiler: node %q: op is required", topologyNode.ID)
	}

	bindMethod, err := op.BindMethod(compiler.registry)

	if err != nil {
		return nil, fmt.Errorf("compiler: node %q: %w", topologyNode.ID, err)
	}

	node := &ir.Node{
		Kind:       ir.KindNode,
		Name:       topologyNode.ID,
		Op:         op,
		BindMethod: bindMethod,
	}

	weightSpec, err := weightSpecForNode(topologyNode)

	if err != nil {
		return nil, err
	}

	if weightSpec == nil {
		return node, nil
	}

	weight, err := compiler.attachWeight(topologyNode.ID, weightSpec, tokenIndex)

	if err != nil {
		return nil, err
	}

	if weight != nil {
		node.Weight = weight
	}

	return node, nil
}

func (compiler *Compiler) attachWeight(
	nodeID string,
	weightSpec *ast.WeightSpec,
	tokenIndex map[string]types.Token,
) (*ir.Weight, error) {
	tensorName := weightSpec.Weight

	if tensorName == "" {
		tensorName = nodeID + ".weight"
	}

	weightToken, ok := tokenIndex[tensorName]

	if !ok {
		return nil, fmt.Errorf("compiler: node %q: missing checkpoint tensor %q", nodeID, tensorName)
	}

	weight := &ir.Weight{
		TensorName: tensorName,
		Tensor:     weightToken,
	}

	if weightSpec.Bias != "" {
		biasToken, biasOK := tokenIndex[weightSpec.Bias]

		if !biasOK {
			return nil, fmt.Errorf("compiler: node %q: missing bias tensor %q", nodeID, weightSpec.Bias)
		}

		weight.BiasName = weightSpec.Bias
		weight.Bias = biasToken

		return weight, nil
	}

	biasName := strings.TrimSuffix(tensorName, ".weight") + ".bias"

	if biasToken, biasOK := tokenIndex[biasName]; biasOK {
		weight.BiasName = biasName
		weight.Bias = biasToken
	}

	return weight, nil
}
