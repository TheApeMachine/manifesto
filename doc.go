/*
Package manifest compiles Hugging Face model repositories and YAML runtime
programs into manifest graph IR, compute IR, and bound weights.

Sub-packages:

  - ast: typed IR for recipes, topology, programs, and weight bindings
  - catalog: recipe and block lookup from an io/fs.FS
  - expand: extends/overrides, repeat unrolling, variable interpolation
  - hfmodular: Hugging Face modular_*.py → recipe YAML transpiler
  - ir: manifest graph IR → pkg/backend/compute/ir
  - lower: topology AST → manifest graph IR
  - parse: YAML loading and include resolution
  - registry: architecture class name → recipe mapping
  - resolve: Hub repo discovery (model_index.json, config.json)
  - runtime: program executor and backend interface
  - weights: SafeTensors loading and graph binding

The root package exposes Compiler as the single entry point. Callers supply a
catalog.FS and resolve.Hub when constructing the compiler.
*/
package manifest
