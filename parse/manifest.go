package parse

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

/*
ManifestDocument preserves a manifest as a generic YAML object.
*/
type ManifestDocument struct {
	Kind     string
	Name     string
	Category string
	Include  map[string]any
	Raw      map[string]any
}

/*
Manifest parses any manifest shape without imposing runtime/model boundaries.
*/
func (parser *Parser) Manifest(data []byte) (*ManifestDocument, error) {
	raw := make(map[string]any)

	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse manifest yaml: %w", err)
	}

	document := &ManifestDocument{
		Raw: raw,
	}

	if value, ok := raw["kind"].(string); ok {
		document.Kind = value
	}

	if value, ok := raw["name"].(string); ok {
		document.Name = value
	}

	if value, ok := raw["category"].(string); ok {
		document.Category = value
	}

	if include, ok := raw["include"].(map[string]any); ok {
		document.Include = include
	}

	if include, ok := raw["includes"].(map[string]any); ok && len(document.Include) == 0 {
		document.Include = include
	}

	return document, nil
}
