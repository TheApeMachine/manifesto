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
Ops with bind.variants require BindMethodForInputCount instead.
*/
func (op Op) BindMethod(registry *OperationRegistry) (string, error) {
	return op.BindMethodForInputCount(registry, 0)
}

/*
BindMethodForInputCount resolves bind.method or the matching bind.variants
entry for one concrete input arity.
*/
func (op Op) BindMethodForInputCount(registry *OperationRegistry, inputCount int) (string, error) {
	if registry == nil {
		return "", fmt.Errorf("operation %q: registry is required", op)
	}

	schema, ok := registry.Lookup(op)

	if !ok {
		return "", fmt.Errorf("operation %q: no schema registered", op)
	}

	if schema.Bind == nil {
		return "", fmt.Errorf("operation %q: bind.method is required", op)
	}

	method, err := resolveBindMethod(schema.Bind, inputCount)

	if err != nil {
		return "", fmt.Errorf("operation %q: %w", op, err)
	}

	return method, nil
}

func resolveBindMethod(bind *asset.Bind, inputCount int) (string, error) {
	if bind == nil {
		return "", fmt.Errorf("bind is required")
	}

	for _, variant := range bind.Variants {
		if variant.When.InputCount == 0 {
			continue
		}

		if variant.When.InputCount != inputCount {
			continue
		}

		if strings.TrimSpace(variant.Method) == "" {
			return "", fmt.Errorf("bind variant for %d input(s) has no method", inputCount)
		}

		return variant.Method, nil
	}

	if strings.TrimSpace(bind.Method) != "" {
		return bind.Method, nil
	}

	if len(bind.Variants) > 0 {
		return "", fmt.Errorf("no bind variant for %d input(s)", inputCount)
	}

	return "", fmt.Errorf("bind.method is required")
}

/*
HasBind reports whether an operation schema resolves to a backend method
for the given input count.
*/
func (registry *OperationRegistry) HasBind(op Op, inputCount int) bool {
	if registry == nil {
		return false
	}

	schema, ok := registry.Lookup(op)

	if !ok || schema.Bind == nil {
		return false
	}

	_, err := resolveBindMethod(schema.Bind, inputCount)

	return err == nil
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

/*
ForEach invokes fn for every registered operation schema.
*/
func (registry *OperationRegistry) ForEach(fn func(op Op, schema asset.Schema)) {
	if registry == nil || fn == nil {
		return
	}

	for op, schema := range registry.schemas {
		fn(op, schema)
	}
}
