package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/tensor"
)

/*
CallGraph executes one graph module with program-session plans and backend wiring.
*/
func (session *ProgramSession) CallGraph(
	ctx context.Context,
	graphName string,
	inputs map[string]any,
	stateOutputs map[string]bool,
) (GraphCallResult, error) {
	if session == nil {
		return GraphCallResult{}, fmt.Errorf("runtime session: session is required")
	}

	graph, ok := session.graphs[graphName]

	if !ok {
		return GraphCallResult{}, fmt.Errorf("runtime session: unknown graph %q", graphName)
	}

	computeGraph, ok := session.compute[graphName]

	if !ok || computeGraph == nil {
		return GraphCallResult{}, fmt.Errorf("runtime session: missing compute graph %q", graphName)
	}

	if session.backend == nil {
		return GraphCallResult{}, fmt.Errorf("runtime session: backend is required")
	}

	return session.backend.CallGraph(ctx, GraphCallRequest{
		GraphName:    graphName,
		Graph:        graph,
		Compute:      computeGraph,
		Plan:         session.plans[graphName],
		Inputs:       inputs,
		StateOutputs: stateOutputs,
	})
}

/*
RunSteps executes a step prefix of the compiled program.
*/
func (session *ProgramSession) RunSteps(
	ctx context.Context,
	steps []ast.Step,
	initial map[string]any,
) error {
	if session == nil {
		return fmt.Errorf("runtime session: session is required")
	}

	executor := NewExecutor(ExecutorOptions{
		Backend:        session.backend,
		Host:           session.host,
		State:          session.state,
		StateMemory:    session.stateMemory,
		ExecutionDType: session.executionDType,
		Plans:          session.plans,
		Stdin:          session.stdin,
		InitialValues:  initial,
	})

	computeAny := make(map[string]any, len(session.compute))

	for name, graph := range session.compute {
		computeAny[name] = graph
	}

	program := &ast.Program{Steps: steps}

	return executor.Run(ctx, program, session.graphs, computeAny)
}

/*
StateStore returns the session state store.
*/
func (session *ProgramSession) StateStore() *StateStore {
	if session == nil {
		return nil
	}

	return session.state
}

/*
StateMemory returns the backend used for resident state tensors.
*/
func (session *ProgramSession) StateMemory() tensor.Backend {
	if session == nil {
		return nil
	}

	return session.stateMemory
}

