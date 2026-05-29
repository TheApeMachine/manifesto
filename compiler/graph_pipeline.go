package compiler

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/typer"
	"github.com/theapemachine/manifesto/types"
	"github.com/theapemachine/manifesto/weights"
)

/*
GraphCompileOptions configures the shared typer → optimizer → codegen pipeline.
*/
type GraphCompileOptions struct {
	OperationRegistry *types.OperationRegistry
	TyperOptions      typer.Options
	SkipTyper         bool
	OptimizerOptions  optimizer.Options
	SkipOptimizer     bool
	CodegenOptions    codegen.EmitOptions
	SkipCodegen       bool
	WeightParser      types.Parser
	WeightMap         map[string]string
}

/*
CompiledGraph is the lowered ast.Graph plus a scheduling DAG built from the
final graph after all compiler passes.
*/
type CompiledGraph struct {
	Graph        *ast.Graph
	ComputeGraph *dag.Graph
}

/*
CompileGraph runs the full graph compilation pipeline on an already-lowered
ast.Graph. When WeightParser is set, checkpoint tensors are bound before
typing so weight PortTypes participate in unification.
*/
func CompileGraph(graph *ast.Graph, options GraphCompileOptions) (*CompiledGraph, error) {
	if graph == nil {
		return nil, fmt.Errorf("compiler: graph is required")
	}

	if options.OperationRegistry == nil {
		return nil, fmt.Errorf("compiler: operation registry is required")
	}

	if options.WeightParser != nil {
		binder := weights.NewBinder()

		if _, err := binder.Bind(graph, options.WeightParser, options.WeightMap); err != nil {
			return nil, fmt.Errorf("compiler: bind weights: %w", err)
		}
	}

	if !options.SkipTyper {
		if _, err := typer.Run(graph, options.TyperOptions); err != nil {
			return nil, err
		}
	}

	if !options.SkipOptimizer {
		if _, err := optimizer.Run(graph, options.OptimizerOptions); err != nil {
			return nil, err
		}

		if !options.SkipTyper {
			if _, err := typer.Run(graph, options.TyperOptions); err != nil {
				return nil, err
			}
		}
	}

	if !options.SkipCodegen {
		if _, err := codegen.AttachKernels(graph, options.CodegenOptions); err != nil {
			return nil, err
		}
	}

	if err := validateGraphOps(graph, options.OperationRegistry); err != nil {
		return nil, err
	}

	computeGraph, err := BuildDAGFromGraph(graph)

	if err != nil {
		return nil, err
	}

	return &CompiledGraph{
		Graph:        graph,
		ComputeGraph: computeGraph,
	}, nil
}
