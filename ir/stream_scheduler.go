package ir

import "fmt"

/*
StreamScheduleOptions configures the §4.4 DAG scheduler.

MaxStreams caps the number of distinct hardware streams the partitioner
will create. ARCHITECTURE.md §5.2 maps StreamID to native primitives
(CUDA streams, Metal command queues, XLA inter-stream deps); each
backend has a hardware-imposed limit on how many in-flight streams are
useful. Setting MaxStreams to 1 forces serial single-stream execution,
which is the safe default when the executor is not yet wired for
multi-stream dispatch.

When MaxStreams is 0 or negative, the partitioner uses the topology's
own width as the cap — every parallel branch gets its own stream.
*/
type StreamScheduleOptions struct {
	MaxStreams int32
}

/*
ScheduleStreams runs the stream-partitioning pass per ARCHITECTURE.md
§5.2. After this returns, every Node.StreamID is assigned and every
node whose dependencies span multiple streams carries SyncBarriers
(SyncEvent entries with Wait=true and the producer's stream).

Algorithm:

 1. Walk nodes in topological order (the topology's Nodes slice is
    assumed already-sorted by the recipe builder; this pass does not
    re-sort).
 2. For each node, look up the StreamID of every predecessor (the
    nodes producing its input ports).
 3. If all predecessors are on the same stream and that stream's most
    recent assignment was this node's most recent predecessor, this
    node continues on that stream — a serial chain.
 4. Otherwise this node opens a new stream (subject to MaxStreams).
 5. For every predecessor on a DIFFERENT stream than the assigned one,
    emit a SyncEvent{Wait: true, StreamID: predecessor's stream}.
 6. Also emit a SyncEvent{Wait: false} on the producer's stream at
    each cross-stream consumption point so the consumer's wait has
    something to wait on. (Recorded on the producer node, not the
    consumer.)

Edge identification uses shared Port pointers: a node's predecessor is
the (earlier-indexed) node whose Outputs slice contains the same Port
pointer as one of this node's Inputs.

Returns an error when MaxStreams is positive but the topology cannot
be partitioned within that cap (i.e., the parallel width exceeds the
limit and we cannot serialize without re-arranging the topological
order, which this pass doesn't do).
*/
func ScheduleStreams(topology *Topology, options StreamScheduleOptions) error {
	if topology == nil {
		return fmt.Errorf("scheduler: topology is required")
	}

	if len(topology.Nodes) == 0 {
		return nil
	}

	maxStreams := options.MaxStreams
	if maxStreams < 1 {
		maxStreams = int32(len(topology.Nodes))
	}

	producerOf := buildProducerIndex(topology)
	streamLastNode := make(map[int32]int32) // streamID → last node ID assigned to it
	nextStreamID := int32(0)
	nextEventID := int32(1)

	for nodeIndex, node := range topology.Nodes {
		if node == nil {
			continue
		}

		// Ensure the node has an ID so SyncEvents can reference it.
		if node.ID == 0 {
			node.ID = int32(nodeIndex + 1)
		}

		predecessorStreams := collectPredecessorStreams(node, topology.Nodes, producerOf)

		assignedStream := pickStream(
			node,
			predecessorStreams,
			streamLastNode,
			&nextStreamID,
			maxStreams,
		)

		if assignedStream < 0 {
			return fmt.Errorf(
				"scheduler: node %q (id=%d) cannot fit within MaxStreams=%d; "+
					"topology has more parallel width than the cap allows",
				node.Name, node.ID, maxStreams,
			)
		}

		node.StreamID = assignedStream
		streamLastNode[assignedStream] = node.ID

		// For each predecessor on a different stream, emit a wait on
		// this node, and a corresponding signal on the producer.
		for predecessorStream, predecessorNodeID := range predecessorStreams {
			if predecessorStream == assignedStream {
				continue
			}

			eventID := nextEventID
			nextEventID++

			node.SyncBarriers = append(node.SyncBarriers, SyncEvent{
				EventID:  eventID,
				StreamID: predecessorStream,
				Wait:     true,
			})

			// Attach the signal to the producer node.
			producerNode := findNodeByID(topology.Nodes, predecessorNodeID)

			if producerNode != nil {
				producerNode.SyncBarriers = append(producerNode.SyncBarriers, SyncEvent{
					EventID:  eventID,
					StreamID: predecessorStream,
					Wait:     false,
				})
			}
		}
	}

	return nil
}

/*
buildProducerIndex returns a map from Port pointer to the index of the
node in topology.Nodes that produces it (i.e., has the port in its
Outputs). Ports that no node produces (graph inputs, weights) are
absent from the map; callers treat absence as "this is a session-init
input, not a runtime dependency."
*/
func buildProducerIndex(topology *Topology) map[*Port]int {
	index := make(map[*Port]int)

	for nodeIndex, node := range topology.Nodes {
		if node == nil {
			continue
		}

		for _, port := range node.Outputs {
			if port == nil {
				continue
			}

			if _, exists := index[port]; !exists {
				index[port] = nodeIndex
			}
		}
	}

	return index
}

/*
collectPredecessorStreams returns the unique set of (StreamID → most
recent producer NodeID) pairs covering all of `node`'s input ports
that have a producer in the topology. Input ports without a producer
(session-init inputs) contribute nothing; SyncEvents are not needed
for them because they're already resident at workspace-init time.
*/
func collectPredecessorStreams(
	node *Node,
	nodes []*Node,
	producerOf map[*Port]int,
) map[int32]int32 {
	streams := make(map[int32]int32)

	for _, port := range node.Inputs {
		if port == nil {
			continue
		}

		producerIndex, found := producerOf[port]
		if !found {
			continue
		}

		producer := nodes[producerIndex]

		if producer == nil {
			continue
		}

		// Keep the most-recent producer per stream so we know who to
		// emit the signal on later.
		streams[producer.StreamID] = producer.ID
	}

	return streams
}

/*
pickStream chooses a StreamID for `node` given its predecessors'
streams. The rule:

  - If the node has exactly one predecessor stream AND the most recent
    node assigned to that stream is one of `node`'s predecessors, the
    node continues that stream (serial chain).
  - Otherwise the node opens a new stream, subject to MaxStreams. If
    the limit forbids a new stream, the node reuses the first
    predecessor stream and the additional predecessors become waits.

Returns -1 if no valid assignment exists (only possible when
MaxStreams=0 and the node has no predecessors, which buildProducerIndex
already prevents — guarded for completeness).
*/
func pickStream(
	node *Node,
	predecessorStreams map[int32]int32,
	streamLastNode map[int32]int32,
	nextStreamID *int32,
	maxStreams int32,
) int32 {
	if len(predecessorStreams) == 0 {
		// Root node (or session-init-only inputs): allocate a fresh
		// stream when possible, otherwise reuse stream 0.
		if *nextStreamID < maxStreams {
			assigned := *nextStreamID
			*nextStreamID++
			return assigned
		}

		return 0
	}

	if len(predecessorStreams) == 1 {
		var onlyStream int32
		var onlyPredecessor int32
		for stream, predecessor := range predecessorStreams {
			onlyStream = stream
			onlyPredecessor = predecessor
		}

		// Continue the serial chain only if this is the very next
		// node after the most recent on that stream. Otherwise the
		// stream has already been used by a parallel branch sibling.
		if streamLastNode[onlyStream] == onlyPredecessor {
			return onlyStream
		}

		// Open a new stream if the cap allows.
		if *nextStreamID < maxStreams {
			assigned := *nextStreamID
			*nextStreamID++
			return assigned
		}

		// Serialize onto the predecessor's stream; the wait still
		// gets emitted but the schedule becomes effectively serial
		// on that stream.
		return onlyStream
	}

	// Multiple predecessor streams: this is a merge point. Pick the
	// lowest-numbered predecessor stream as the consumer's stream,
	// and the others become waits.
	var lowestStream int32 = -1
	for stream := range predecessorStreams {
		if lowestStream < 0 || stream < lowestStream {
			lowestStream = stream
		}
	}

	return lowestStream
}

func findNodeByID(nodes []*Node, id int32) *Node {
	for _, node := range nodes {
		if node != nil && node.ID == id {
			return node
		}
	}

	return nil
}
