package typer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
Options configures one typer.Run.
*/
type Options struct {
	// MaxAdaptorRounds caps how many times the typer will run
	// Infer → SynthesizeAdaptors → Infer to converge. Adaptor synthesis
	// can introduce new edges whose types need to be unified, so we
	// loop until no resolvable edge errors remain or the cap fires.
	// Zero means use the default (3).
	MaxAdaptorRounds int

	// DisableSynthesis turns off adaptor insertion entirely. Used by
	// tests that want to assert on the raw EdgeError output.
	DisableSynthesis bool
}

/*
Stats reports counters from one end-to-end typer pass.
*/
type Stats struct {
	Infer      InferStats
	Synthesis  SynthesisStats
	Rounds     int
	Unresolved []EdgeError
}

/*
Run is the public entry point. It runs Infer, then SynthesizeAdaptors
on the resulting edge errors, then Infer again to type the newly
inserted nodes — looping until no resolvable edge errors remain.

Returns an error only for hard failures (rank mismatch, semantic-kind
mismatch, constraint violation) that can't be resolved by an adaptor
node. The returned Stats.Unresolved lists those edge errors so callers
can render diagnostics.
*/
func Run(graph *ast.Graph, options Options) (Stats, error) {
	if graph == nil {
		return Stats{}, fmt.Errorf("typer: graph is required")
	}

	maxRounds := options.MaxAdaptorRounds

	if maxRounds <= 0 {
		maxRounds = 3
	}

	stats := Stats{}

	for round := 0; round < maxRounds; round++ {
		inferStats, edgeErrors, err := Infer(graph)

		if err != nil {
			return stats, err
		}

		stats.Infer = inferStats
		stats.Rounds = round + 1

		if len(edgeErrors) == 0 {
			return stats, nil
		}

		if options.DisableSynthesis {
			stats.Unresolved = edgeErrors
			return stats, nil
		}

		synthStats, unresolved, err := SynthesizeAdaptors(graph, edgeErrors)

		if err != nil {
			return stats, err
		}

		stats.Synthesis.CastsInserted += synthStats.CastsInserted
		stats.Synthesis.TransposesInserted += synthStats.TransposesInserted
		stats.Synthesis.ReshapesInserted += synthStats.ReshapesInserted
		stats.Synthesis.Unresolved = synthStats.Unresolved

		if len(unresolved) > 0 {
			stats.Unresolved = unresolved
			return stats, &HardFailure{Errors: unresolved}
		}

		// No new adaptors were inserted but edge errors remain — re-
		// running Infer would just produce the same errors. Stop now.
		if synthStats.CastsInserted+synthStats.TransposesInserted+synthStats.ReshapesInserted == 0 {
			stats.Unresolved = edgeErrors
			return stats, nil
		}
	}

	return stats, fmt.Errorf("typer: did not converge after %d rounds", maxRounds)
}

/*
HardFailure aggregates edge errors that can't be resolved by adaptor
synthesis. Callers can type-assert on this to render a unified
diagnostic with every offending edge at once.
*/
type HardFailure struct {
	Errors []EdgeError
}

func (failure *HardFailure) Error() string {
	if len(failure.Errors) == 0 {
		return "typer: hard failure (no edges)"
	}

	if len(failure.Errors) == 1 {
		return failure.Errors[0].Error()
	}

	return fmt.Sprintf("typer: %d unresolved edge errors (first: %s)", len(failure.Errors), failure.Errors[0].Error())
}
