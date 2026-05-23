package types

import "iter"

/*
Parser walks one in-memory archive and yields normalized tokens.
*/
type Parser interface {
	Generate() (iter.Seq[Token], error)
}
