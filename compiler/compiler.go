package compiler

import (
	"fmt"
	"sort"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/types"
)

/*
Compiler builds manifest IR from checkpoint tokens produced by a Parser.
*/
type Compiler struct {
	parser          types.Parser
	operationLookup *OperationLookup
}

/*
NewCompiler constructs a Compiler over one archive parser.
*/
func NewCompiler(parser types.Parser) (*Compiler, error) {
	if parser == nil {
		return nil, fmt.Errorf("compiler: parser is required")
	}

	return &Compiler{
		parser:          parser,
		operationLookup: NewOperationLookup(),
	}, nil
}

/*
Compile builds Project → Architecture → Topology → Node from parser tokens.
*/
func (compiler *Compiler) Compile() (*ir.Project, error) {
	tokenIndex, err := ir.NewTokenIndex(compiler.parser)

	if err != nil {
		return nil, fmt.Errorf("compiler: index tokens: %w", err)
	}

	project, err := compiler.buildProject(tokenIndex)

	if err != nil {
		return nil, fmt.Errorf("compiler: build project: %w", err)
	}

	return project, nil
}

/*
buildProject constructs Project → Architecture → Topology → Node from a token index.
*/
func (compiler *Compiler) buildProject(tokenIndex *ir.TokenIndex) (*ir.Project, error) {
	if tokenIndex == nil {
		return nil, fmt.Errorf("build project: token index is required")
	}

	nodeDrafts, err := compiler.indexNodeDrafts(tokenIndex)

	if err != nil {
		return nil, err
	}

	nodes := make([]*ir.Node, 0, len(nodeDrafts))
	nodeNames := make([]string, 0, len(nodeDrafts))

	for nodeName := range nodeDrafts {
		nodeNames = append(nodeNames, nodeName)
	}

	sort.Strings(nodeNames)

	for _, nodeName := range nodeNames {
		node, buildErr := nodeDrafts[nodeName].Node(compiler.operationLookup)

		if buildErr != nil {
			return nil, buildErr
		}

		nodes = append(nodes, node)
	}

	projectName, architectureName := compiler.projectNames(tokenIndex)

	return &ir.Project{
		Kind:        ir.KindResearchProject,
		Name:        projectName,
		Description: "compiled from checkpoint tokens",
		Architecture: &ir.Architecture{
			Kind:        ir.KindArchitecture,
			Name:        architectureName,
			Description: "compiled from checkpoint tokens",
			Topology: &ir.Topology{
				Kind:        ir.KindTopology,
				Name:        "topology",
				Description: "compiled from checkpoint tokens",
				Nodes:       nodes,
			},
		},
	}, nil
}

/*
projectNames reads project and architecture names from checkpoint metadata.
*/
func (compiler *Compiler) projectNames(tokenIndex *ir.TokenIndex) (projectName string, architectureName string) {
	projectName = "checkpoint"
	architectureName = "model"

	if metadata, ok := tokenIndex.Metadata("model_type"); ok && metadata.Value != "" {
		architectureName = metadata.Value
	}

	if metadata, ok := tokenIndex.Metadata("format"); ok && metadata.Value != "" {
		projectName = metadata.Value
	}

	return projectName, architectureName
}

/*
indexNodeDrafts groups checkpoint tensor tokens by node prefix.
*/
func (compiler *Compiler) indexNodeDrafts(tokenIndex *ir.TokenIndex) (map[string]*NodeDraft, error) {
	nodeDrafts := make(map[string]*NodeDraft)

	for tensorName, token := range tokenIndex.Tensors() {
		nodeName, paramSuffix, ok := compiler.operationLookup.SplitNodeParam(tensorName)

		if !ok {
			continue
		}

		nodeDraft, exists := nodeDrafts[nodeName]

		if !exists {
			nodeDraft = NewNodeDraft(nodeName, len(token.Shape))
			nodeDrafts[nodeName] = nodeDraft
		}

		nodeDraft.AbsorbParam(tensorName, token, paramSuffix)
	}

	if len(nodeDrafts) == 0 {
		return nil, fmt.Errorf("build project: no checkpoint tensors mapped to nodes")
	}

	return nodeDrafts, nil
}
