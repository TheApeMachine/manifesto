package ir

import "time"

/*
Topology is the IR of a model architecture.
*/
type Topology struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Nodes       []*Node
	Edges       []*Edge
}
