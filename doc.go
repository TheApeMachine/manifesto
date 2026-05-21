/*
Package manifesto is the module root for the manifest compiler and runtime.

Import sub-packages directly:

  - ast: typed IR for recipes, topology, programs, and weight bindings
  - asset: embedded YAML templates
  - catalog: recipe and block lookup from an io/fs.FS
  - compiler: YAML program and Hub repo compilation
  - expand: extends/overrides, repeat unrolling, variable interpolation
  - ir: manifest graph IR → compute IR
  - lower: topology AST → manifest graph IR
  - parse: YAML loading into typed program and block AST
  - registry: architecture class name → recipe mapping
  - resolve: Hub repo discovery (model_index.json, config.json)
  - runtime: program executor and backend interface
  - weights: SafeTensors loading and graph binding
*/
package manifesto
