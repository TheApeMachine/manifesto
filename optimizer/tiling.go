package optimizer

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

/*
TileAttribute is the ast.GraphNode.Attributes key under which cache-tiling
configuration is attached to heavy ops (Matmul, Conv2D). Codegen reads this
attribute when emitting the kernel.
*/
const TileAttribute = "cache_tiling"

/*
TileConfig is the cache-tiling shape attached to one heavy op.
*/
type TileConfig struct {
	// Rows, Inner, Cols are the per-tile dimensions for a Matmul. For
	// other op kinds the meaningful axes are documented per-op below.
	Rows  int
	Inner int
	Cols  int

	// L1Footprint is the byte size estimate of one tile, rounded up to the
	// workspace alignment. Used by the static memory planner to avoid
	// stepping outside L1 once it lands.
	L1Footprint int64
}

/*
TilingStats summarizes one cache-tiling pass.
*/
type TilingStats struct {
	TilesAttached int
}

/*
TileTargetConfig describes the host hardware's cache hierarchy. The
optimizer reads from this struct rather than detecting at runtime so
deterministic test reproduction is possible.
*/
type TileTargetConfig struct {
	L1Bytes int64
	L2Bytes int64
	Vector  int // SIMD vector lane count in elements (e.g. 16 for AVX-512 f32).
}

/*
DefaultTileTarget is a conservative CPU profile: 32 KiB L1, 256 KiB L2,
8-lane vectors. It's a sane fallback for AVX2/NEON; AVX-512 hosts should
override Vector at the caramba layer once CPUID detection lands.
*/
func DefaultTileTarget() TileTargetConfig {
	return TileTargetConfig{
		L1Bytes: 32 * 1024,
		L2Bytes: 256 * 1024,
		Vector:  8,
	}
}

/*
Tile attaches TileAttribute annotations to every matmul/conv node on graph.
This is the §4.4 ("Cache-Tiling Transformations") pass.

The implementation only attaches metadata — the actual tiled loops are
emitted by codegen. The tile sizes are chosen so that one tile's working
set fits in L1, leaving room for at least the two operand panels and the
accumulator block.

The pass is a no-op on graphs that already carry a tile config (allows
manual override via YAML).
*/
func Tile(graph *ast.Graph, target TileTargetConfig) (TilingStats, error) {
	if graph == nil {
		return TilingStats{}, fmt.Errorf("optimizer: graph is required")
	}

	stats := TilingStats{}

	for _, node := range graph.Nodes {
		if node == nil {
			continue
		}

		if !isHeavyOp(node.Op) {
			continue
		}

		if node.Attributes == nil {
			node.Attributes = make(map[string]any)
		}

		if _, alreadySet := node.Attributes[TileAttribute]; alreadySet {
			continue
		}

		tile := pickTile(target)

		node.Attributes[TileAttribute] = tile
		stats.TilesAttached++
	}

	return stats, nil
}

func isHeavyOp(op string) bool {
	switch op {
	case "math.matmul", "projection.linear",
		"convolution.conv1d", "convolution.conv2d", "convolution.conv3d":
		return true
	default:
		return false
	}
}

/*
pickTile selects (rows, inner, cols) so that the L1 footprint of one tile
(plus its two operand panels) stays under L1Bytes / 4. The "/ 4" reserve
leaves room for the accumulator block and for the kernel's stack frame.

Element size assumed at 4 bytes (f32 / bf16-as-f32 accumulator). Codegen
adjusts when targeting smaller dtypes — but tile sizes computed for f32
remain safe for any narrower dtype, just suboptimal.
*/
func pickTile(target TileTargetConfig) TileConfig {
	budget := target.L1Bytes / 4

	if budget < 256 {
		budget = 256
	}

	// Aim for square-ish tiles aligned to the SIMD vector width.
	vector := int64(target.Vector)

	if vector <= 0 {
		vector = 4
	}

	side := vector

	for {
		footprint := (side*side + 2*side) * 4

		if footprint > budget {
			break
		}

		side += vector
	}

	side -= vector

	if side < vector {
		side = vector
	}

	return TileConfig{
		Rows:        int(side),
		Inner:       int(side),
		Cols:        int(side),
		L1Footprint: side * side * 4,
	}
}
