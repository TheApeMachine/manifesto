package typer

import (
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/types"
)

var (
	registrySpecsOnce sync.Once
	registrySpecs     map[string]OpSpec
	registrySpecsErr  error
)

func loadRegistrySpecs() {
	registry, err := types.NewOperationRegistry()

	if err != nil {
		registrySpecsErr = fmt.Errorf("typer: operation registry: %w", err)
		return
	}

	registrySpecs = make(map[string]OpSpec)

	registry.ForEach(func(op types.Op, schema asset.Schema) {
		opName := strings.TrimSpace(string(op))

		if opName == "" {
			return
		}

		if _, exists := specTable[opName]; exists {
			return
		}

		registrySpecs[opName] = opSpecFromSchema(schema)
	})
}

func opSpecFromSchema(schema asset.Schema) OpSpec {
	inputCount := len(schema.Inputs)

	if inputCount == 0 {
		return OpSpec{
			Inputs:        []ir.PortType{anyTensor()},
			OutputDeriver: deriveSameAsFirstInput(ir.SemanticGeneric),
		}
	}

	inputs := make([]ir.PortType, inputCount)

	for index := range inputs {
		inputs[index] = anyTensor()
	}

	spec := OpSpec{
		Inputs: inputs,
	}

	switch inputCount {
	case 1:
		spec.OutputDeriver = deriveSameAsFirstInput(ir.SemanticGeneric)
	default:
		spec.OutputDeriver = deriveSameAsFirstInput(ir.SemanticGeneric)
	}

	return spec
}
