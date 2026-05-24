package ir

/*
SyncEvent is one hardware-synchronization point between two streams.
The scheduler (ARCHITECTURE.md §5.2) emits SyncEvents on a Node when
the node depends on outputs from a different stream, or when the node
produces outputs another stream will consume.

The executor maps SyncEvents to native primitives:
  - CUDA: cudaEventRecord (signal) / cudaStreamWaitEvent (wait)
  - Metal: MTLEvent encodeSignalEvent / encodeWaitForEvent
  - XLA: PJRT inter-stream dependencies through the HLO scheduler
*/
type SyncEvent struct {
	// EventID is a session-scope unique identifier the executor uses
	// to resolve this event to the native primitive it allocated at
	// session init. Pre-resolved into a flat handle table; never
	// looked up by map at dispatch time per §5.2.
	EventID int32
	// StreamID identifies the hardware stream that signals or waits on
	// this event.
	StreamID int32
	// Wait toggles whether this event is a pre-dispatch wait (true) or
	// a post-dispatch signal (false).
	Wait bool
}
