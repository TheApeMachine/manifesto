package registry

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
)

/*
Registry resolves Hugging Face class names to catalog recipes.
*/
type Registry struct {
	catalog catalog.Catalog
	entries map[string]ast.RegistryEntry
}

/*
NewRegistry constructs a Registry backed by the supplied catalog.
*/
func NewRegistry(catalogInstance catalog.Catalog) (*Registry, error) {
	entries, err := catalogInstance.Registry()

	if err != nil {
		return nil, err
	}

	return &Registry{
		catalog: catalogInstance,
		entries: entries,
	}, nil
}

/*
Lookup returns the registry entry for a Hugging Face architecture class.
*/
func (registry *Registry) Lookup(className string) (ast.RegistryEntry, error) {
	entry, ok := registry.entries[className]

	if !ok {
		return ast.RegistryEntry{}, fmt.Errorf("registry: unknown architecture %q", className)
	}

	return entry, nil
}

/*
Recipe loads and returns the recipe referenced by a class name.
*/
func (registry *Registry) Recipe(className string) (*ast.Recipe, error) {
	entry, err := registry.Lookup(className)

	if err != nil {
		return nil, err
	}

	return registry.catalog.LoadRecipe(entry.Include)
}
