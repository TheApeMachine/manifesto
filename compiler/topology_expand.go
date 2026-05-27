package compiler

import (
	"fmt"
	"maps"
	"strconv"
	"strings"

	"github.com/theapemachine/manifesto/ast"
)

/*
ExpandedTopology is the flat result of materializing every `repeat` template
in a topology. Inputs are passed through unchanged.
*/
type ExpandedTopology struct {
	Inputs   []string
	Outputs  map[string]string
	Nodes    []ast.Node
	Bindings map[string]int64
}

/*
expandTopology materializes every `control.repeat` node into its inlined
instances. Each instance substitutes the loop index variable into IDs,
input refs, output refs, and weight strings. Nested repeats are not yet
supported and return an error.
*/
func expandTopology(topology *ast.Topology) (*ExpandedTopology, error) {
	out := &ExpandedTopology{
		Inputs:   append([]string(nil), topology.Inputs...),
		Outputs:  cloneOutputRefs(topology.Outputs),
		Bindings: maps.Clone(topology.Bindings),
	}

	for _, node := range topology.Nodes {
		if !isRepeatNode(node) {
			out.Nodes = append(out.Nodes, node)
			continue
		}

		instances, err := expandRepeatNode(node)

		if err != nil {
			return nil, err
		}

		out.Nodes = append(out.Nodes, instances...)
	}

	return out, nil
}

func cloneOutputRefs(outputs map[string]string) map[string]string {
	return maps.Clone(outputs)
}

func isRepeatNode(node ast.Node) bool {
	op := strings.ToLower(strings.TrimSpace(node.Op))

	return op == "control.repeat" || (node.Repeat != nil && op == "")
}

func expandRepeatNode(node ast.Node) ([]ast.Node, error) {
	count, err := repeatCount(node.Repeat)

	if err != nil {
		return nil, fmt.Errorf("expand repeat %q: %w", node.ID, err)
	}

	offset, err := repeatOffset(node.Offset)

	if err != nil {
		return nil, fmt.Errorf("expand repeat %q: %w", node.ID, err)
	}

	indexVar := strings.TrimSpace(node.Index)

	if indexVar == "" {
		indexVar = "i"
	}

	if len(node.Template) == 0 {
		return nil, fmt.Errorf("expand repeat %q: template body is empty", node.ID)
	}

	out := make([]ast.Node, 0, count*len(node.Template))

	for index := 0; index < count; index++ {
		for _, templateNode := range node.Template {
			if isRepeatNode(templateNode) {
				return nil, fmt.Errorf(
					"expand repeat %q: nested repeats are not supported",
					node.ID,
				)
			}

			out = append(out, substituteNode(templateNode, indexVar, index, offset))
		}
	}

	return out, nil
}

func substituteNode(node ast.Node, indexVar string, index int, offset int) ast.Node {
	indexLiteral := strconv.Itoa(index)
	prevLiteral := strconv.Itoa(index)
	nextLiteral := strconv.Itoa(index + 1)
	offsetLiteral := strconv.Itoa(index + offset)
	prevOffsetLiteral := strconv.Itoa(index + offset - 1)
	nextOffsetLiteral := strconv.Itoa(index + offset + 1)

	replacer := newIndexReplacer(
		indexVar,
		indexLiteral,
		prevLiteral,
		nextLiteral,
		offsetLiteral,
		prevOffsetLiteral,
		nextOffsetLiteral,
	)

	substituted := ast.Node{
		ID:     replacer.Replace(node.ID),
		Op:     node.Op,
		Config: substituteConfig(node.Config, replacer),
	}

	substituted.In = make([]string, 0, len(node.In))

	for _, value := range node.In {
		substituted.In = append(substituted.In, replacer.Replace(value))
	}

	substituted.Out = make([]string, 0, len(node.Out))

	for _, value := range node.Out {
		substituted.Out = append(substituted.Out, replacer.Replace(value))
	}

	if node.Weights != nil {
		substituted.Weights = &ast.WeightSpec{
			Weight:     replacer.Replace(node.Weights.Weight),
			Bias:       replacer.Replace(node.Weights.Bias),
			SliceAxis:  node.Weights.SliceAxis,
			SliceStart: substituteConfigValue(node.Weights.SliceStart, replacer),
			SliceEnd:   substituteConfigValue(node.Weights.SliceEnd, replacer),
		}
	}

	return substituted
}

func substituteConfig(config map[string]any, replacer *strings.Replacer) map[string]any {
	if len(config) == 0 {
		return nil
	}

	out := make(map[string]any, len(config))

	for key, value := range config {
		out[key] = substituteConfigValue(value, replacer)
	}

	return out
}

func substituteConfigValue(value any, replacer *strings.Replacer) any {
	switch typed := value.(type) {
	case string:
		replaced := replacer.Replace(typed)

		if parsed, err := strconv.Atoi(strings.TrimSpace(replaced)); err == nil {
			return parsed
		}

		return replaced
	case []any:
		out := make([]any, 0, len(typed))

		for _, item := range typed {
			out = append(out, substituteConfigValue(item, replacer))
		}

		return out
	case map[string]any:
		out := make(map[string]any, len(typed))

		for key, item := range typed {
			out[key] = substituteConfigValue(item, replacer)
		}

		return out
	default:
		return value
	}
}

func repeatOffset(raw any) (int, error) {
	if raw == nil {
		return 0, nil
	}

	return repeatCount(raw)
}

func repeatCount(raw any) (int, error) {
	switch typed := raw.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		count, err := strconv.Atoi(strings.TrimSpace(typed))

		if err != nil {
			return 0, fmt.Errorf("invalid repeat count %q: %w", typed, err)
		}

		return count, nil
	case nil:
		return 0, fmt.Errorf("repeat count is required")
	default:
		return 0, fmt.Errorf("unsupported repeat count type %T", raw)
	}
}

/*
newIndexReplacer builds a string replacer for the loop-index substitution
forms used in topology templates:

  - ${i}        → current iteration index (e.g. "h_${i}"   → "h_3")
  - ${i+1}      → next iteration index    (e.g. "h_${i+1}" → "h_4")
  - ${i-1}      → previous iteration      (e.g. "h_${i-1}" → "h_2")
  - ${next_i}   → alias for ${i+1}, common in the LlamaForCausalLM and
    similar HF-loader architecture templates so authors can write the
    residual chain as `out: h_${next_i}` without the inline arithmetic.
  - ${prev_i}   → alias for ${i-1}, symmetric counterpart.
*/
func newIndexReplacer(
	indexVar string,
	indexLiteral string,
	prevLiteral string,
	nextLiteral string,
	offsetLiteral string,
	prevOffsetLiteral string,
	nextOffsetLiteral string,
) *strings.Replacer {
	_ = prevLiteral

	openBrace := "${"
	closeBrace := "}"

	indexToken := openBrace + indexVar + closeBrace
	nextToken := openBrace + indexVar + "+1" + closeBrace
	prevToken := openBrace + indexVar + "-1" + closeBrace
	nextAlias := openBrace + "next_" + indexVar + closeBrace
	prevAlias := openBrace + "prev_" + indexVar + closeBrace
	offsetAlias := openBrace + "offset_" + indexVar + closeBrace
	nextOffsetAlias := openBrace + "next_offset_" + indexVar + closeBrace
	prevOffsetAlias := openBrace + "prev_offset_" + indexVar + closeBrace

	prevLiteralValue := strconv.Itoa(safePrev(indexLiteral))

	return strings.NewReplacer(
		indexToken, indexLiteral,
		nextToken, nextLiteral,
		prevToken, prevLiteralValue,
		nextAlias, nextLiteral,
		prevAlias, prevLiteralValue,
		offsetAlias, offsetLiteral,
		nextOffsetAlias, nextOffsetLiteral,
		prevOffsetAlias, prevOffsetLiteral,
	)
}

func safePrev(literal string) int {
	value, err := strconv.Atoi(literal)

	if err != nil {
		return 0
	}

	return value - 1
}
