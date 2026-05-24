package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/compiler"
)

/*
CompileProgramFromAsset parses and compiles one program manifest path from the asset FS.
*/
func CompileProgramFromAsset(
	ctx context.Context,
	programPath string,
	cacheDir string,
) (*compiler.CompileOutput, error) {
	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return nil, fmt.Errorf("runtime orchestrator: read program %q: %w", programPath, err)
	}

	manifestCompiler, err := compiler.NewProgramCompiler(
		compiler.NewPool(catalog.NewFS(asset.TemplateFS())),
	)

	if err != nil {
		return nil, fmt.Errorf("runtime orchestrator: new compiler: %w", err)
	}

	output, err := manifestCompiler.CompileAssets(ctx, compiler.CompileInput{
		ProgramYAML: programYAML,
		CacheDir:    cacheDir,
	}, asset.TemplateFS())

	if err != nil {
		return nil, fmt.Errorf("runtime orchestrator: compile assets: %w", err)
	}

	return output, nil
}
