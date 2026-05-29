package ir

import (
	"fmt"
	"iter"

	"github.com/theapemachine/manifesto/types"
)

/*
TokenIndex maps checkpoint keys from a serialized archive to normalized tokens.
*/
type TokenIndex struct {
	tensors  map[string]types.Token
	metadata map[string]types.Token
}

/*
NewTokenIndex builds a lookup map from every token a Parser yields.
*/
func NewTokenIndex(parser types.Parser) (*TokenIndex, error) {
	if parser == nil {
		return nil, fmt.Errorf("token index: parser is required")
	}

	sequence := parser.Generate()

	tokenIndex := &TokenIndex{
		tensors:  make(map[string]types.Token),
		metadata: make(map[string]types.Token),
	}

	for token := range sequence {
		switch token.Kind {
		case types.KindTensor:
			tokenIndex.tensors[token.Name] = token
		case types.KindMetadata:
			tokenIndex.metadata[token.Name] = token
		}
	}

	return tokenIndex, nil
}

/*
Tensor returns the token for a checkpoint tensor name.
*/
func (tokenIndex *TokenIndex) Tensor(name string) (types.Token, bool) {
	if tokenIndex == nil || name == "" {
		return types.Token{}, false
	}

	token, ok := tokenIndex.tensors[name]

	return token, ok
}

/*
Metadata returns the token for an archive metadata entry.
*/
func (tokenIndex *TokenIndex) Metadata(name string) (types.Token, bool) {
	if tokenIndex == nil || name == "" {
		return types.Token{}, false
	}

	token, ok := tokenIndex.metadata[name]

	return token, ok
}

/*
Tensors iterates indexed checkpoint tensors in undefined order.
*/
func (tokenIndex *TokenIndex) Tensors() iter.Seq2[string, types.Token] {
	return func(yield func(string, types.Token) bool) {
		if tokenIndex == nil {
			return
		}

		for name, token := range tokenIndex.tensors {
			if !yield(name, token) {
				return
			}
		}
	}
}

/*
TensorCount returns the number of indexed checkpoint tensors.
*/
func (tokenIndex *TokenIndex) TensorCount() int {
	if tokenIndex == nil {
		return 0
	}

	return len(tokenIndex.tensors)
}

/*
MetadataCount returns the number of indexed metadata entries.
*/
func (tokenIndex *TokenIndex) MetadataCount() int {
	if tokenIndex == nil {
		return 0
	}

	return len(tokenIndex.metadata)
}
