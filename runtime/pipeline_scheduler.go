package runtime

import (
	"fmt"
	"maps"

	"github.com/theapemachine/manifesto/ast"
)

func schedulersFromPipelineIncludes(program *ast.Program) (map[string]*FlowMatchEulerDiscrete, error) {
	if program == nil {
		return nil, nil
	}

	schedulers := make(map[string]*FlowMatchEulerDiscrete)

	for _, includeObject := range program.IncludeObjects {
		document, ok := includeObject.(map[string]any)

		if !ok {
			continue
		}

		kind, _ := document["kind"].(string)

		if kind != "Pipeline" {
			continue
		}

		pipelineSchedulers, err := schedulersFromPipelineDocument(document, program.Variables)

		if err != nil {
			return nil, err
		}

		maps.Copy(schedulers, pipelineSchedulers)
	}

	return schedulers, nil
}

func schedulersFromPipelineDocument(
	document map[string]any,
	variables map[string]any,
) (map[string]*FlowMatchEulerDiscrete, error) {
	components, ok := nestedPipelineMap(document, "system", "components")

	if !ok {
		return nil, fmt.Errorf("pipeline include: missing system.components")
	}

	schedulers := make(map[string]*FlowMatchEulerDiscrete)

	for componentName, rawComponent := range components {
		component, ok := rawComponent.(map[string]any)

		if !ok {
			continue
		}

		className, _ := component["class_name"].(string)

		if schedulerTypeFromClassName(className) == "" {
			continue
		}

		hubConfig, ok := component["config"].(map[string]any)

		if !ok {
			return nil, fmt.Errorf("pipeline scheduler %q: missing config", componentName)
		}

		schedulerConfig, err := schedulerConfigFromHub(hubConfig, variables)

		if err != nil {
			return nil, fmt.Errorf("pipeline scheduler %q: %w", componentName, err)
		}

		scheduler, err := NewFlowMatchEulerDiscrete(schedulerConfig)

		if err != nil {
			return nil, fmt.Errorf("pipeline scheduler %q: %w", componentName, err)
		}

		schedulers[componentName] = scheduler
	}

	if len(schedulers) == 0 {
		return nil, fmt.Errorf("pipeline include: no scheduler component found")
	}

	return schedulers, nil
}

func schedulerTypeFromClassName(className string) string {
	switch className {
	case "FlowMatchEulerDiscreteScheduler":
		return "flow_match_euler_discrete"
	default:
		return ""
	}
}

func nestedPipelineMap(document map[string]any, path ...string) (map[string]any, bool) {
	current := any(document)

	for _, segment := range path {
		next, ok := current.(map[string]any)

		if !ok {
			return nil, false
		}

		current, ok = next[segment]

		if !ok {
			return nil, false
		}
	}

	result, ok := current.(map[string]any)

	return result, ok
}

func schedulerNameFromConfig(config map[string]any) string {
	name, ok := config["scheduler"].(string)

	if ok && name != "" {
		return name
	}

	return "scheduler"
}
