package parse

import (
	"fmt"
	"maps"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
BlockModel is a caramba model manifest block (topology + hub runtime metadata).
*/
type BlockModel struct {
	Outputs []blockPort `yaml:"outputs"`
	System  blockSystem `yaml:"system"`
}

type blockPort struct {
	Name string `yaml:"name"`
}

type blockSystem struct {
	Runtime  blockRuntimeConfig `yaml:"runtime"`
	Topology topologySection    `yaml:"topology"`
}

type blockRuntimeConfig struct {
	Type        string         `yaml:"type"`
	Backend     string         `yaml:"backend"`
	Model       hubSource      `yaml:"model"`
	Tokenizer   hubSource      `yaml:"tokenizer"`
	TextEncoder componentRef   `yaml:"text_encoder"`
	Transformer componentRef   `yaml:"transformer"`
	VAE         componentRef   `yaml:"vae"`
	Generation  map[string]any `yaml:"generation"`
}

type hubSource struct {
	Source   string `yaml:"source"`
	RepoType string `yaml:"repo_type"`
	Revision string `yaml:"revision"`
	File     string `yaml:"file"`
}

type componentRef struct {
	Source       string         `yaml:"source"`
	Path         string         `yaml:"path"`
	File         string         `yaml:"file"`
	Architecture string         `yaml:"architecture"`
	Manifest     string         `yaml:"manifest"`
	Config       map[string]any `yaml:"config"`
}

/*
BlockModel parses a model block manifest from YAML bytes.
*/
func BlockModelFromYAML(data []byte) (*BlockModel, error) {
	block := &BlockModel{}

	if err := yaml.Unmarshal(data, block); err != nil {
		return nil, fmt.Errorf("parse block model yaml: %w", err)
	}

	return block, nil
}

/*
PrimaryRepoID returns the main Hugging Face repo for this block.
*/
func (block *BlockModel) PrimaryRepoID() string {
	if block == nil {
		return ""
	}

	if block.System.Runtime.Model.Source != "" {
		return block.System.Runtime.Model.Source
	}

	if block.System.Topology.FromSafeTensors != nil {
		if source, ok := block.System.Topology.FromSafeTensors["source"].(string); ok {
			return source
		}
	}

	return ""
}

/*
WeightSubfolder returns the component directory declared by the model weight
file, if the manifest points at a repository subfolder.
*/
func (block *BlockModel) WeightSubfolder() string {
	if block == nil {
		return ""
	}

	directory := filepath.Dir(block.System.Runtime.Model.File)

	if directory == "." {
		return ""
	}

	return directory
}

/*
TopologyAST returns inline topology nodes when present.
*/
func (block *BlockModel) TopologyAST() (*ast.Topology, error) {
	if block == nil {
		return nil, fmt.Errorf("parse block model: input is required")
	}

	if len(block.System.Topology.Nodes) > 0 {
		return &ast.Topology{
			Inputs:   block.System.Topology.Inputs,
			Outputs:  block.outputMap(),
			Nodes:    block.System.Topology.Nodes,
			Bindings: block.System.Topology.Bindings,
		}, nil
	}

	return nil, fmt.Errorf("parse block model: no topology nodes")
}

func (block *BlockModel) outputMap() map[string]string {
	outputs := maps.Clone(block.System.Topology.Outputs)

	if outputs == nil && len(block.Outputs) > 0 {
		outputs = make(map[string]string, len(block.Outputs))
	}

	for _, output := range block.Outputs {
		if output.Name == "" {
			continue
		}

		if _, exists := outputs[output.Name]; exists {
			continue
		}

		outputs[output.Name] = output.Name
	}

	return outputs
}

/*
FromSafeTensorsSpec returns the from_safetensors section when declared.
*/
func (block *BlockModel) FromSafeTensorsSpec() map[string]any {
	if block == nil {
		return nil
	}

	return block.System.Topology.FromSafeTensors
}
