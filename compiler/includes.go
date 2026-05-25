package compiler

import (
	"context"
	"fmt"
	"strings"
)

/*
IncludeSource is one entry from a program manifest's include block.
Name is the local include name (e.g. "model"), Source is the raw URI or
asset path (e.g. "hf://meta-llama/Llama-3.2-1B-Instruct" or
"model/diffusion/flux-2-klein-4b.yml").
*/
type IncludeSource struct {
	Name   string
	Source string
}

/*
IncludeResolver loads block YAML for one program include directive. Hosts
implement this interface to bridge manifesto compilation to external sources
(HuggingFace Hub, local files, embedded templates). Manifest never imports a
concrete hub client.
*/
type IncludeResolver interface {
	/*
		ResolveInclude returns block YAML for the given include source. The
		returned bytes must parse as a parse.BlockModel (canonical
		`system.runtime`/`system.topology` shape).
	*/
	ResolveInclude(ctx context.Context, include IncludeSource) ([]byte, error)
}

/*
IncludeSchemeHF is the URI scheme manifesto programs use to reference a
HuggingFace repository (e.g. "hf://meta-llama/Llama-3.2-1B-Instruct").
*/
const IncludeSchemeHF = "hf://"

/*
ParseHFReference splits an "hf://" URI into the repo id and an optional
component subpath following a "#" separator. For example:

	"hf://black-forest-labs/FLUX.2-klein-4B#transformer"
	→ "black-forest-labs/FLUX.2-klein-4B", "transformer", true
*/
func ParseHFReference(source string) (repoID, component string, ok bool) {
	if !strings.HasPrefix(source, IncludeSchemeHF) {
		return "", "", false
	}

	body := strings.TrimPrefix(source, IncludeSchemeHF)

	if body == "" {
		return "", "", false
	}

	if hashIndex := strings.Index(body, "#"); hashIndex >= 0 {
		return body[:hashIndex], body[hashIndex+1:], true
	}

	return body, "", true
}

/*
IsHFReference reports whether a source is an "hf://" URI.
*/
func IsHFReference(source string) bool {
	_, _, ok := ParseHFReference(source)
	return ok
}

/*
ResolverError wraps a resolver error with the include name for context.
*/
type ResolverError struct {
	Include string
	Cause   error
}

func (err *ResolverError) Error() string {
	return fmt.Sprintf("compiler: include %q: %v", err.Include, err.Cause)
}

func (err *ResolverError) Unwrap() error {
	return err.Cause
}
