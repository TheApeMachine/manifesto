package compiler

import (
	"context"
	"fmt"
	"io/fs"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/ir/dag"
	"github.com/theapemachine/manifesto/parse"
)

/*
CompileInput is the host-provided context for one program compilation.
*/
type CompileInput struct {
	ProgramYAML []byte
	CacheDir    string
}

/*
CompileOutput is the compiled program and named compute graphs.
*/
type CompileOutput struct {
	Program       *ast.Program
	Graphs        map[string]*ast.Graph
	ComputeGraphs map[string]*dag.Graph
}

/*
Pool resolves manifest assets and Hugging Face repositories for program compilation.
*/
type Pool struct {
	catalog catalog.Catalog
}

/*
NewPool constructs a compiler asset pool.
*/
func NewPool(catalogInstance catalog.Catalog) *Pool {
	return &Pool{catalog: catalogInstance}
}

/*
ProgramCompiler parses and compiles manifest program YAML into runtime IR.
Graph lowering is filled in as the ARCHITECTURE.md pipeline lands in manifesto.
*/
type ProgramCompiler struct {
	pool   *Pool
	parser *parse.Parser
}

/*
NewProgramCompiler constructs a program compiler from host-provided dependencies.
*/
func NewProgramCompiler(pool *Pool) (*ProgramCompiler, error) {
	if pool == nil {
		return nil, fmt.Errorf("compiler: asset pool is required")
	}

	return &ProgramCompiler{
		pool:   pool,
		parser: parse.NewParser(),
	}, nil
}

/*
CompileAssets parses one program manifest and prepares runtime outputs.
*/
func (programCompiler *ProgramCompiler) CompileAssets(
	ctx context.Context,
	input CompileInput,
	assetFS fs.FS,
) (*CompileOutput, error) {
	_ = ctx
	_ = assetFS
	_ = programCompiler.pool
	_ = input.CacheDir

	if len(input.ProgramYAML) == 0 {
		return nil, fmt.Errorf("compiler: program yaml is required")
	}

	program, err := programCompiler.parser.Program(input.ProgramYAML)

	if err != nil {
		return nil, fmt.Errorf("compiler: parse program: %w", err)
	}

	return &CompileOutput{
		Program:       program,
		Graphs:        make(map[string]*ast.Graph),
		ComputeGraphs: make(map[string]*dag.Graph),
	}, nil
}
