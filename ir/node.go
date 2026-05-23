package ir

import "time"

/*
Node is one operation in the topology.
*/
type Node struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	Operation   Operation
	Weight      *Weight
	Inputs      []*Port
	Outputs     []*Port
}
