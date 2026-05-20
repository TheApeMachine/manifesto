package tensor

import (
	"context"
	"sync"
)

/*
GradFn is a node on the autograd tape. The forward kernel that
produced an output tensor records a GradFn that, given the upstream
gradient (a tensor with the same shape and dtype as the forward
output), computes the gradient with respect to each input.

Tape.Backward (below) drives the walk. Each GradFn knows its inputs
and its output; the tape uses Output() to look up the accumulated
gradient that flows into Backward, and uses Inputs() to know where
to accumulate the returned per-input gradients.
*/
type GradFn interface {
	// Backward computes the gradient with respect to each input
	// given the upstream gradient. The returned slice has the same
	// length and order as Inputs.
	Backward(ctx context.Context, upstream Tensor) ([]Tensor, error)

	// Inputs returns the tensors this op consumed during the
	// forward pass. The tape walks these to propagate gradients.
	Inputs() []Tensor

	// Output returns the tensor this op produced during the
	// forward pass. The tape uses it to fetch the upstream
	// gradient before invoking Backward.
	Output() Tensor

	// Name returns a human-readable op name for debugging.
	Name() string
}

/*
Tape records GradFn nodes in forward execution order. Backward walks
in reverse, accumulating gradients on each tensor that has
RequiresGrad set.
*/
type Tape struct {
	mu    sync.Mutex
	nodes []GradFn
}

/*
NewTape returns an empty Tape.
*/
func NewTape() *Tape {
	return &Tape{}
}

/*
Record appends a GradFn to the tape. Forward kernels call this when
any of their inputs has RequiresGrad set; the output tensor inherits
RequiresGrad=true and its GradFn field points at the recorded node.
*/
func (tape *Tape) Record(node GradFn) {
	tape.mu.Lock()
	tape.nodes = append(tape.nodes, node)
	tape.mu.Unlock()
}

/*
Length returns the number of recorded ops.
*/
func (tape *Tape) Length() int {
	tape.mu.Lock()
	defer tape.mu.Unlock()

	return len(tape.nodes)
}

/*
Backward walks the tape in reverse, computing gradients and
accumulating them into each input's Grad slot.

Seed: the output tensor's gradient must already be set (by the
caller) before invoking Backward. For scalar-loss training the
seed is typically a same-shape ones tensor; for ranked loss outputs
the seed is whatever the upstream optimizer hands in.

Per AGENTS.md §1 and §3.11 of TENSOR_BACKEND_REWRITE.md, this is the
generic driver; per-op Backward implementations live alongside each
forward kernel and must register a backward kernel for each accepted
forward signature. Missing coverage surfaces as ErrBackwardNotImplemented
at record time (when the forward kernel runs with RequiresGrad
inputs), not at backward time.
*/
func (tape *Tape) Backward(ctx context.Context, output Tensor) error {
	tape.mu.Lock()
	nodes := append([]GradFn(nil), tape.nodes...)
	tape.mu.Unlock()

	if len(nodes) == 0 {
		return nil
	}

	for index := len(nodes) - 1; index >= 0; index-- {
		node := nodes[index]
		upstream, err := node.Output().Grad()

		if err != nil {
			return err
		}

		grads, err := node.Backward(ctx, upstream)

		if err != nil {
			return err
		}

		if len(grads) != len(node.Inputs()) {
			return ErrShapeMismatch
		}

		for inputIndex, input := range node.Inputs() {
			if !input.RequiresGrad() {
				continue
			}

			if err := accumulateGrad(input, grads[inputIndex]); err != nil {
				return err
			}
		}
	}

	return nil
}

/*
accumulateGrad adds delta into target.Grad() in place, allocating a
gradient tensor on first contact. For host tensors this is a straight
elementwise add through Float64Native (the bridge dtype until per-dtype
backward kernels land). Backends with their own gradient accumulation
provide their own SetGrad path; the generic helper here is the
fallback.
*/
func accumulateGrad(target, delta Tensor) error {
	host, ok := target.(*HostTensor)

	if !ok {
		return ErrLayoutUnsupported
	}

	deltaHost, ok := delta.(*HostTensor)

	if !ok {
		return ErrLayoutUnsupported
	}

	existing := host.grad.Load()

	if existing == nil {
		// First gradient contribution: take delta wholesale.
		host.grad.Store(deltaHost)
		return nil
	}

	existingView, err := existing.Float64Native()

	if err != nil {
		return err
	}

	deltaView, err := deltaHost.Float64Native()

	if err != nil {
		return err
	}

	if len(existingView) != len(deltaView) {
		return ErrShapeMismatch
	}

	for index := range existingView {
		existingView[index] += deltaView[index]
	}

	return nil
}

/*
Clear empties the tape. Used between training steps.
*/
func (tape *Tape) Clear() {
	tape.mu.Lock()
	tape.nodes = nil
	tape.mu.Unlock()
}

/*
SetHostGrad seeds a gradient on a host tensor. Used to initialize the
output's gradient before calling Tape.Backward, and by per-op
backward kernels that want to set a gradient explicitly. Returns
ErrLayoutUnsupported if target is not a *HostTensor; device-tensor
gradient seeding goes through backend-specific paths that Phase 11
adds alongside each device's autograd implementation.
*/
func SetHostGrad(target Tensor, grad Tensor) error {
	host, ok := target.(*HostTensor)

	if !ok {
		return ErrLayoutUnsupported
	}

	gradHost, ok := grad.(*HostTensor)

	if !ok {
		return ErrLayoutUnsupported
	}

	host.grad.Store(gradHost)
	return nil
}

/*
SimpleGradFn is a concrete GradFn helper for kernels that don't need
custom storage. Forward kernels populate the fields and append to the
tape via tape.Record.
*/
type SimpleGradFn struct {
	OpName    string
	InputList []Tensor
	OutTensor Tensor
	BackFn    func(ctx context.Context, upstream Tensor) ([]Tensor, error)
}

/*
Backward dispatches to the configured BackFn.
*/
func (fn *SimpleGradFn) Backward(ctx context.Context, upstream Tensor) ([]Tensor, error) {
	if fn.BackFn == nil {
		return nil, ErrBackwardNotImplemented
	}

	return fn.BackFn(ctx, upstream)
}

/*
Inputs returns the recorded input tensors.
*/
func (fn *SimpleGradFn) Inputs() []Tensor { return fn.InputList }

/*
Output returns the recorded output tensor.
*/
func (fn *SimpleGradFn) Output() Tensor { return fn.OutTensor }

/*
Name returns the op name.
*/
func (fn *SimpleGradFn) Name() string { return fn.OpName }

var _ GradFn = (*SimpleGradFn)(nil)
