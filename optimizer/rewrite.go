package optimizer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
RewriteStats summarizes one algebraic-rewrite pass.
*/
type RewriteStats struct {
	IdentitiesRemoved int
	ScalesFolded      int
}

/*
Rewrite applies algebraic identities to graph in place. Today it covers two
patterns that show up routinely in HF-generated topologies:

 1. Identity elimination — a node whose op is "shape.identity" or whose
    output equals one of its inputs and has no parameters can be deleted;
    consumers are rewired to the surviving producer.

 2. Fold-scale-into-Linear — a math.mul by a scalar literal feeding a
    projection.linear can be absorbed into the projection by scaling the
    weight matrix at compile time. The actual weight rewrite happens in
    the codegen stage (we only flag the linear node with a
    "fold_scale" attribute); rewriting weight tensors here would require
    materializing the safetensors data.

This pass is intentionally small. Additional rewrites (constant folding,
commutative reordering, dead-code elimination over fused subgraphs) belong
in dedicated files so each rule stays grep-friendly.
*/
func Rewrite(graph *ast.Graph) (RewriteStats, error) {
	if graph == nil {
		return RewriteStats{}, fmt.Errorf("optimizer: graph is required")
	}

	stats := RewriteStats{}

	rewriteIdentities(graph, &stats)
	flagScaleFolds(graph, &stats)

	return stats, nil
}

func rewriteIdentities(graph *ast.Graph, stats *RewriteStats) {
	// Build a redirection table: producer name → replacement name.
	redirect := make(map[string]string)
	survivors := make([]*ast.GraphNode, 0, len(graph.Nodes))

	for _, node := range graph.Nodes {
		if !isIdentityNode(node) {
			survivors = append(survivors, node)
			continue
		}

		if len(node.Inputs) != 1 {
			survivors = append(survivors, node)
			continue
		}

		// The identity's output becomes an alias for its single input.
		redirect[node.ID] = node.Inputs[0]
		stats.IdentitiesRemoved++
	}

	if len(redirect) == 0 {
		return
	}

	// Apply redirects transitively (an identity of an identity collapses
	// to the original producer).
	for source := range redirect {
		current := source

		for {
			next, ok := redirect[current]

			if !ok {
				break
			}

			current = next
		}

		redirect[source] = current
	}

	for _, node := range survivors {
		for index, inputName := range node.Inputs {
			if replacement, ok := redirect[inputName]; ok {
				node.Inputs[index] = replacement
			}
		}
	}

	for outputName, ref := range graph.Outputs {
		if replacement, ok := redirect[ref]; ok {
			graph.Outputs[outputName] = replacement
		}
	}

	graph.Nodes = survivors
}

func isIdentityNode(node *ast.GraphNode) bool {
	if node == nil {
		return false
	}

	if node.Op == "shape.identity" || node.Op == "control.identity" {
		return true
	}

	return false
}

/*
flagScaleFolds marks projection.linear nodes whose only input is a math.mul
by a scalar constant. The codegen stage absorbs the scale by multiplying
the projection's weight matrix at load time.

This pass only flags — it never mutates weights — because the weight
tensors live in safetensors and aren't materialized at compile time.
*/
func flagScaleFolds(graph *ast.Graph, stats *RewriteStats) {
	producers := make(map[string]*ast.GraphNode)

	for _, node := range graph.Nodes {
		producers[node.ID] = node
	}

	consumerCount := make(map[string]int)

	for _, node := range graph.Nodes {
		for _, inputName := range node.Inputs {
			consumerCount[inputName]++
		}
	}

	for _, node := range graph.Nodes {
		if node.Op != "projection.linear" {
			continue
		}

		if len(node.Inputs) != 1 {
			continue
		}

		producer, ok := producers[node.Inputs[0]]

		if !ok || producer.Op != "math.mul" {
			continue
		}

		if consumerCount[producer.ID] != 1 {
			continue
		}

		scalar, hasScalar := producer.Attributes["scalar"]

		if !hasScalar {
			continue
		}

		if node.Attributes == nil {
			node.Attributes = make(map[string]any)
		}

		node.Attributes["fold_scale"] = scalar
		stats.ScalesFolded++
	}
}
