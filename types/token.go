package types

import "github.com/theapemachine/manifesto/dtype"

/*
Kind distinguishes metadata entries from tensor declarations in an archive index.
*/
type Kind uint8

const (
	KindMetadata Kind = iota + 1
	KindTensor
)

/*
Token is one entry from a serialized weight archive before payload bytes are read.

For KindTensor, Name is the full checkpoint key. Topology manifests bind graph
nodes to these names through weight entries, optionally with slice ranges when
one fused tensor feeds multiple nodes.

For KindMetadata, Name is the metadata key and Value is the metadata string.
Shape, Precision, and Span are unused.
*/
type Token struct {
	Kind      Kind
	Name      string
	Value     string
	Shape     []int64
	Precision dtype.DType
	Span      Span
}

/*
Span locates tensor payload bytes relative to the start of the archive data buffer.
*/
type Span struct {
	Offset int64
	Length int64
}
