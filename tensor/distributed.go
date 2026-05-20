package tensor

import "github.com/theapemachine/manifesto/dtype"

/*
ShardingMesh describes a topology of devices that a DistributedTensor
spans. Devices is a flat list in mesh-flatten order (last axis fastest);
Shape gives the per-axis sizes; AxisNames optionally labels axes for
operator readability ("data", "model", "pipeline").
*/
type ShardingMesh struct {
	Devices   []Location
	Shape     []int
	AxisNames []string
}

/*
ShardingSpec describes how a logical tensor of rank R partitions
across a ShardingMesh. PerDim has length R.
*/
type ShardingSpec struct {
	PerDim []DimSharding
}

/*
DimSharding describes how a single logical dimension partitions
across the mesh. Replicated tensors duplicate the dimension on every
device along the named mesh axis; sharded tensors split the dimension
evenly. ShardAxis is the index into ShardingMesh.Shape; -1 means
"this dim does not interact with the mesh."
*/
type DimSharding struct {
	Replicated bool
	ShardAxis  int
}

/*
DistributedTensor is a logical tensor whose storage is split across
multiple Locations under a ShardingSpec. Phase 10 implements the
concrete types and the collective ops that move data between shards.
*/
type DistributedTensor interface {
	LogicalShape() Shape
	DType() dtype.DType
	Layout() Layout
	Mesh() ShardingMesh
	Sharding() ShardingSpec

	// Shards returns the per-mesh-device physical tensors. Indexing
	// is mesh-flatten order: device at coords (i, j) on a 2-D mesh
	// with shape [I, J] is shards[i*J + j].
	Shards() []Tensor

	// LocalShard returns the shard for the device this process is
	// driving. Returns an error if the process has not been
	// associated with a specific device through the process-group
	// bootstrap (see §3.10).
	LocalShard() (Tensor, error)

	Close() error
}
