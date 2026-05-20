package ast

/*
Recipe is a composable architecture definition. Recipes extend parent recipes
or blocks and declare config bindings and weight maps used during expansion.
*/
type Recipe struct {
	Name      string
	Extends   string
	Overrides map[string]any
	Config    map[string]Binding
	WeightMap map[string]string
	Topology  *Topology
	Inputs    []string
	Outputs   map[string]string
}

/*
Binding resolves a recipe variable from a Hugging Face config field or expression.
*/
type Binding struct {
	Config  string    `yaml:"config,omitempty" json:"config,omitempty"`
	Product []Binding `yaml:"product,omitempty" json:"product,omitempty"`
	Sum     []Binding `yaml:"sum,omitempty" json:"sum,omitempty"`
	Literal any       `yaml:"literal,omitempty" json:"literal,omitempty"`
}

/*
RegistryEntry maps a Hugging Face architecture or pipeline class to a recipe.
*/
type RegistryEntry struct {
	Include   string
	Variables map[string]Binding
}
