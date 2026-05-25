package optimizer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
Options configures one optimizer Run.
*/
type Options struct {
	// DisableConstantFold skips the constant-folding pass.
	DisableConstantFold bool

	// DisableRewrite skips the algebraic-rewrite pass.
	DisableRewrite bool

	// DisableFusion skips the elementwise-fusion pass.
	DisableFusion bool

	// DisableTiling skips the cache-tiling pass.
	DisableTiling bool

	// TileTarget describes the host hardware cache hierarchy. If zero,
	// DefaultTileTarget is used.
	TileTarget TileTargetConfig
}

/*
Stats reports counters from every sub-pass.
*/
type Stats struct {
	ConstantFold ConstantFoldStats
	Rewrite      RewriteStats
	Fusion       FusionStats
	Tiling       TilingStats
}

/*
Run executes the optimizer pipeline on graph in place. The pass order is
fixed (rewrite → fuse → tile) because each pass relies on the previous
pass's structure:

  - Rewrite reshapes the graph (identity elim, scale-into-Linear flags) so
    the fuser sees the post-rewrite node set.
  - Fuse clusters elementwise nodes — fewer, larger nodes — which both
    shrinks subsequent tiling work and ensures the tiler ignores nodes that
    have been absorbed into fused subgraphs.
  - Tile attaches cache-tiling metadata to the remaining heavy ops
    (Matmul / Conv*).

Callers consume the returned Stats for diagnostics; the graph mutation is
the primary effect.
*/
func Run(graph *ast.Graph, options Options) (Stats, error) {
	if graph == nil {
		return Stats{}, fmt.Errorf("optimizer: graph is required")
	}

	stats := Stats{}

	if !options.DisableConstantFold {
		constantFoldStats, err := ConstantFold(graph)

		if err != nil {
			return stats, fmt.Errorf("optimizer: constant fold: %w", err)
		}

		stats.ConstantFold = constantFoldStats
	}

	if !options.DisableRewrite {
		rewriteStats, err := Rewrite(graph)

		if err != nil {
			return stats, fmt.Errorf("optimizer: rewrite: %w", err)
		}

		stats.Rewrite = rewriteStats
	}

	if !options.DisableFusion {
		fusionStats, err := Fuse(graph)

		if err != nil {
			return stats, fmt.Errorf("optimizer: fuse: %w", err)
		}

		stats.Fusion = fusionStats
	}

	if !options.DisableTiling {
		tileTarget := options.TileTarget

		if tileTarget.L1Bytes == 0 {
			tileTarget = DefaultTileTarget()
		}

		tilingStats, err := Tile(graph, tileTarget)

		if err != nil {
			return stats, fmt.Errorf("optimizer: tile: %w", err)
		}

		stats.Tiling = tilingStats
	}

	return stats, nil
}
