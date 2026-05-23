package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/resolve"
	"github.com/theapemachine/manifesto/types"
)

/*
CompileProgramFromAsset parses and compiles one program manifest path from the asset FS.
*/
func CompileProgramFromAsset(
	ctx context.Context,
	programPath string,
	hub resolve.Hub,
	parser func(archive []byte) (types.Parser, error),
	cacheDir string,
) (*compiler.CompileOutput, error) {
	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return nil, fmt.Errorf("runtime orchestrator: read program %q: %w", programPath, err)
	}

	manifestCompiler, err := compiler.NewCompiler(
		ctx,
		compiler.NewPool(catalog.NewFS(asset.TemplateFS()), hub),
		parser,
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
