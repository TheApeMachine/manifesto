package tensor

import "fmt"

/*
State is the lifecycle state of a Tensor. Transitions are enforced by
the implementations of the Tensor interface and by the Backend's
upload path; see §2.8 of TENSOR_BACKEND_REWRITE.md for the contract.

The four states correspond to:

  - StateReady: the tensor's storage is valid and can be read or
    mutated through native views, uploaded, downloaded, migrated.
  - StatePending: an async backend op is in flight against this
    tensor. Host-side view acquisition returns ErrTensorInTransit;
    Close blocks until the op completes or the operation's context
    expires.
  - StateMutating: a native view is outstanding. Uploads, migrations,
    or other state-changing ops return ErrTensorMutating until the
    view's GC cleanup releases the reservation.
  - StateClosed: Close has run. Every operation errors with
    ErrTensorClosed.
*/
type State uint32

const (
	StateReady State = iota
	StatePending
	StateMutating
	StateClosed
)

/*
String returns a human-readable name for the state.
*/
func (state State) String() string {
	switch state {
	case StateReady:
		return "ready"
	case StatePending:
		return "pending"
	case StateMutating:
		return "mutating"
	case StateClosed:
		return "closed"
	}

	return fmt.Sprintf("state(%d)", state)
}
