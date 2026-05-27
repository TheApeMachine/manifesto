package ast

/*
Topology is a fully expanded or partially expanded compute DAG expressed in
manifest primitive operations.
*/
type Topology struct {
	Inputs   []string
	Outputs  map[string]string
	Nodes    []Node
	Bindings map[string]int64
}

/*
Node is one primitive operation in a topology graph.
*/
type Node struct {
	ID       string
	Op       string
	In       []string
	Out      []string
	Config   map[string]any
	Repeat   any
	Index    string
	Offset   any
	Template []Node
	Weights  *WeightSpec
}

/*
WeightSpec binds a node parameter to a SafeTensors tensor, optionally slicing a
fused checkpoint tensor.
*/
type WeightSpec struct {
	Weight     string
	Bias       string
	SliceAxis  string
	SliceStart any
	SliceEnd   any
}
