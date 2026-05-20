package hfmodular

import (
	"fmt"
	"io"
)

/*
Transpiler converts Hugging Face modular_*.py files into manifest recipe YAML.
*/
type Transpiler struct {
	source []byte
}

/*
NewTranspiler constructs a Transpiler for one modular Python source file.
*/
func NewTranspiler(source []byte) *Transpiler {
	return &Transpiler{source: source}
}

/*
Transpile parses a modular Python file and emits recipe YAML bytes.
*/
func (transpiler *Transpiler) Transpile() ([]byte, error) {
	if len(transpiler.source) == 0 {
		return nil, fmt.Errorf("hfmodular transpile: source is required")
	}

	return nil, fmt.Errorf("hfmodular transpile: not implemented")
}

/*
TranspileReader reads all bytes from reader and transpiles them.
*/
func (transpiler *Transpiler) TranspileReader(reader io.Reader) ([]byte, error) {
	source, err := io.ReadAll(reader)

	if err != nil {
		return nil, fmt.Errorf("hfmodular transpile read: %w", err)
	}

	transpiler.source = source

	return transpiler.Transpile()
}
