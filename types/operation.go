package types

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/asset"
)

/*
Op is a manifest operation identifier matching the op: field in
template/operation/*.yml (e.g. "activation.gelu", "projection.linear").
Topology nodes declare Op; the compiler resolves bind.method from the
embedded operation catalog rather than inferring from checkpoint names.
*/
type Op string

/*
String returns the canonical op identifier.
*/
func (op Op) String() string {
	return string(op)
}

/*
BindMethod returns the device.Backend method name from the operation
schema's bind.method field (e.g. "Gelu", "Matmul", "RMSNorm").
*/
func (op Op) BindMethod(registry *OperationRegistry) (string, error) {
	if registry == nil {
		return "", fmt.Errorf("operation %q: registry is required", op)
	}

	schema, ok := registry.Lookup(op)

	if !ok {
		return "", fmt.Errorf("operation %q: no schema registered", op)
	}

	if schema.Bind == nil || schema.Bind.Method == "" {
		return "", fmt.Errorf("operation %q: bind.method is required", op)
	}

	return schema.Bind.Method, nil
}

/*
OperationRegistry indexes embedded operation manifests by op identifier.
*/
type OperationRegistry struct {
	schemas map[Op]asset.Schema
}

/*
NewOperationRegistry loads every operation schema from the embedded
template/operation tree.
*/
func NewOperationRegistry() (*OperationRegistry, error) {
	raw, err := asset.Walk("template/operation")

	if err != nil {
		return nil, fmt.Errorf("operation registry: %w", err)
	}

	schemas := make(map[Op]asset.Schema, len(raw))

	for key, schema := range raw {
		op := Op(strings.TrimSpace(key))

		if op == "" {
			continue
		}

		schemas[op] = schema
	}

	return &OperationRegistry{schemas: schemas}, nil
}

/*
Lookup returns the schema for one manifest op identifier.
*/
func (registry *OperationRegistry) Lookup(op Op) (asset.Schema, bool) {
	if registry == nil {
		return asset.Schema{}, false
	}

	schema, ok := registry.schemas[op]

	return schema, ok
}

/*
Count returns the number of registered operation schemas.
*/
func (registry *OperationRegistry) Count() int {
	if registry == nil {
		return 0
	}

	return len(registry.schemas)
}
