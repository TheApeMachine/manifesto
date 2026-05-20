package ast

import "github.com/theapemachine/manifesto/dtype"

/*
Pipeline describes a multi-component Hugging Face repository discovered from
model_index.json.
*/
type Pipeline struct {
	ClassName  string
	Components map[string]Component
}

/*
Component is one sub-model inside a pipeline repository.
*/
type Component struct {
	Library     string
	ClassName   string
	Subfolder   string
	Config      map[string]any
	WeightFiles []string
}

/*
ModelBundle is the compiled output for one repo or program include.
*/
type ModelBundle struct {
	RepoID     string
	Revision   string
	Pipeline   *Pipeline
	Components map[string]*ComponentGraph
	Tokenizer  *AssetRef
}

/*
ComponentGraph holds the lowered graph and bound weights for one component.
*/
type ComponentGraph struct {
	ClassName      string
	ExecutionDType dtype.DType
	Graph          *Graph
	Weights        map[string]*BoundWeight
}

/*
AssetRef locates a downloaded artifact inside a resolved repository snapshot.
*/
type AssetRef struct {
	RepoID   string
	Revision string
	Path     string
}
