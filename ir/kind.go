package ir

/*
Kind distinguishes metadata entries from tensor declarations in the index.
*/
type Kind uint8

const (
	KindResearchProject Kind = iota + 1
	KindArchitecture
	KindTopology
	KindBlock
	KindNode
	KindEdge
	KindPort
	KindTensor
)
