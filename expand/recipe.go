package expand

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
)

/*
Recipe expands a recipe and its extends chain into a concrete topology.
*/
type Recipe struct {
	catalog catalog.Catalog
}

/*
NewRecipe constructs a Recipe expander backed by the supplied catalog.
*/
func NewRecipe(catalogInstance catalog.Catalog) *Recipe {
	return &Recipe{catalog: catalogInstance}
}

/*
Topology materializes a recipe against a Hugging Face config map.
*/
func (expander *Recipe) Topology(
	recipe *ast.Recipe,
	config map[string]any,
) (*ast.Topology, error) {
	merged, err := expander.mergeChain(recipe)

	if err != nil {
		return nil, err
	}

	if merged.Topology == nil {
		return nil, fmt.Errorf("expand recipe %q: no topology", recipe.Name)
	}

	variables, err := expander.bindConfig(merged.Config, config)

	if err != nil {
		return nil, err
	}

	nodes, err := expander.expandNodes(merged.Topology.Nodes, variables)

	if err != nil {
		return nil, err
	}

	return &ast.Topology{
		Inputs: merged.Topology.Inputs,
		Nodes:  nodes,
	}, nil
}

func (expander *Recipe) mergeChain(recipe *ast.Recipe) (*ast.Recipe, error) {
	if recipe.Extends == "" {
		return recipe, nil
	}

	parent, err := expander.catalog.LoadRecipe(recipe.Extends)

	if err != nil {
		parentBlock, blockErr := expander.catalog.LoadBlock(recipe.Extends)

		if blockErr != nil {
			return nil, fmt.Errorf("expand extends %q: %w", recipe.Extends, err)
		}

		return &ast.Recipe{
			Name:      recipe.Name,
			Extends:   recipe.Extends,
			Topology:  parentBlock,
			Config:    recipe.Config,
			WeightMap: recipe.WeightMap,
			Overrides: recipe.Overrides,
		}, nil
	}

	parentMerged, err := expander.mergeChain(parent)

	if err != nil {
		return nil, err
	}

	return expander.mergeRecipes(parentMerged, recipe), nil
}

func (expander *Recipe) mergeRecipes(base, override *ast.Recipe) *ast.Recipe {
	merged := *base
	merged.Name = override.Name

	if override.Topology != nil {
		merged.Topology = override.Topology
	}

	for key, value := range override.Config {
		if merged.Config == nil {
			merged.Config = make(map[string]ast.Binding)
		}

		merged.Config[key] = value
	}

	for key, value := range override.WeightMap {
		if merged.WeightMap == nil {
			merged.WeightMap = make(map[string]string)
		}

		merged.WeightMap[key] = value
	}

	for key, value := range override.Overrides {
		if merged.Overrides == nil {
			merged.Overrides = make(map[string]any)
		}

		merged.Overrides[key] = value
	}

	return &merged
}
