package parse

import (
	"fmt"

	"gopkg.in/yaml.v3"

	"github.com/theapemachine/manifesto/ast"
)

/*
BlockModel is a caramba model manifest block (topology + hub runtime metadata).
*/
type BlockModel struct {
	System blockSystem `yaml:"system"`
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
TopologyAST returns inline topology nodes when present.
*/
func (block *BlockModel) TopologyAST() (*ast.Topology, error) {
	if block == nil {
		return nil, fmt.Errorf("parse block model: input is required")
	}

	if len(block.System.Topology.Nodes) > 0 {
		return &ast.Topology{
			Inputs: block.System.Topology.Inputs,
			Nodes:  block.System.Topology.Nodes,
		}, nil
	}

	return nil, fmt.Errorf("parse block model: no topology nodes")
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
