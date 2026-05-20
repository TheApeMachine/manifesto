package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
Backend executes manifest graph modules on a host compute target. Device
backends implement this interface outside pkg/manifest.
*/
type Backend interface {
	CallGraph(
		ctx context.Context,
		request GraphCallRequest,
	) (GraphCallResult, error)
}

/*
GraphCallRequest is one graph.call runtime step.
*/
type GraphCallRequest struct {
	GraphName string
	Graph     *ast.Graph
	Inputs    map[string]any
}

/*
GraphCallResult holds graph.call outputs.
*/
type GraphCallResult struct {
	Outputs map[string]any
}

/*
Executor runs a manifest program against a backend.
*/
type Executor struct {
	backend Backend
}

/*
NewExecutor constructs an Executor backed by the supplied runtime backend.
*/
func NewExecutor(backend Backend) *Executor {
	return &Executor{backend: backend}
}

/*
Run executes program steps sequentially.
*/
func (executor *Executor) Run(
	ctx context.Context,
	program *ast.Program,
	graphs map[string]*ast.Graph,
) error {
	if program == nil {
		return fmt.Errorf("runtime execute: program is required")
	}

	values := make(map[string]any)

	for _, step := range program.Steps {
		if err := executor.runStep(ctx, step, graphs, values); err != nil {
			return fmt.Errorf("runtime step %q op %q: %w", step.ID, step.Op, err)
		}
	}

	return nil
}

func (executor *Executor) runStep(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	values map[string]any,
) error {
	switch step.Op {
	case "graph.call":
		return executor.runGraphCall(ctx, step, graphs, values)
	default:
		return fmt.Errorf("unsupported runtime op %q", step.Op)
	}
}

func (executor *Executor) runGraphCall(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	values map[string]any,
) error {
	graphName := step.Graph

	if graphName == "" {
		configured, ok := step.Config["graph"].(string)

		if !ok || configured == "" {
			return fmt.Errorf("graph.call requires graph name")
		}

		graphName = configured
	}

	graph, ok := graphs[graphName]

	if !ok {
		return fmt.Errorf("unknown graph %q", graphName)
	}

	inputs := make(map[string]any, len(step.In))

	for name, ref := range step.In {
		inputs[name] = values[ref]
	}

	result, err := executor.backend.CallGraph(ctx, GraphCallRequest{
		GraphName: graphName,
		Graph:     graph,
		Inputs:    inputs,
	})

	if err != nil {
		return err
	}

	for name, ref := range step.Out {
		values[ref] = result.Outputs[name]
	}

	return nil
}
