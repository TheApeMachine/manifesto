package ir

import "time"

/*
Block is one operation in the topology.
*/
type Block struct {
	Kind        Kind
	Name        string
	Description string
	Created     *time.Time
	Updated     *time.Time
	System      *System
}
