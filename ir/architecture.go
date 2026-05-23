package ir

import "time"

/*
Architecture is the IR of a model architecture.
*/
type Architecture struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Topology    *Topology
}
