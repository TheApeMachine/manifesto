package catalog

import (
	"fmt"
	"io/fs"
	"path"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

const registryPath = "model/architecture/registry.yml"

/*
Catalog resolves recipes, blocks, and the architecture registry from an fs.FS.
Hosts mount embedded templates or an on-disk tree at the FS root.
*/
type Catalog interface {
	Registry() (map[string]ast.RegistryEntry, error)
	LoadRecipe(name string) (*ast.Recipe, error)
	LoadBlock(name string) (*ast.Topology, error)
	ReadRaw(filename string) ([]byte, error)
}

/*
FS implements Catalog over any fs.FS.
Hosts mount embedded templates or an on-disk tree at the FS root.
*/
type FS struct {
	files fs.FS
}

/*
NewFS constructs a catalog over the supplied filesystem.
*/
func NewFS(files fs.FS) *FS {
	return &FS{files: files}
}

func (catalog *FS) Registry() (map[string]ast.RegistryEntry, error) {
	raw, err := fs.ReadFile(catalog.files, registryPath)

	if err != nil {
		return nil, fmt.Errorf("catalog registry: %w", err)
	}

	document := struct {
		Architectures map[string]ast.RegistryEntry `yaml:"architectures"`
	}{}

	if err := yaml.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("catalog registry parse: %w", err)
	}

	return document.Architectures, nil
}

func (catalog *FS) LoadRecipe(name string) (*ast.Recipe, error) {
	filename := recipePath(name)
	raw, err := fs.ReadFile(catalog.files, filename)

	if err != nil {
		return nil, fmt.Errorf("catalog recipe %q: %w", name, err)
	}

	recipe := &ast.Recipe{Name: name}

	if err := yaml.Unmarshal(raw, recipe); err != nil {
		return nil, fmt.Errorf("catalog recipe parse %q: %w", name, err)
	}

	return recipe, nil
}

func (catalog *FS) LoadBlock(name string) (*ast.Topology, error) {
	filename := blockPath(name)
	raw, err := fs.ReadFile(catalog.files, filename)

	if err != nil {
		return nil, fmt.Errorf("catalog block %q: %w", name, err)
	}

	topology := &ast.Topology{}

	if err := yaml.Unmarshal(raw, topology); err != nil {
		return nil, fmt.Errorf("catalog block parse %q: %w", name, err)
	}

	return topology, nil
}

func (catalog *FS) ReadRaw(filename string) ([]byte, error) {
	raw, err := fs.ReadFile(catalog.files, filename)

	if err != nil {
		return nil, fmt.Errorf("catalog read %q: %w", filename, err)
	}

	return raw, nil
}

func recipePath(name string) string {
	return path.Join("model", "recipe", normalizeName(name)+".yml")
}

func blockPath(name string) string {
	return path.Join("model", "block", normalizeName(name)+".yml")
}

func normalizeName(name string) string {
	return strings.NewReplacer(".", "/", "_", "/").Replace(name)
}
