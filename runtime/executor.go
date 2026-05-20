package runtime

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"sort"
	"strings"

	"github.com/theapemachine/manifesto/ast"
)

/*
Backend executes manifest graph modules on a host compute target.
*/
type Backend interface {
	CallGraph(
		ctx context.Context,
		request GraphCallRequest,
	) (GraphCallResult, error)
}

/*
GraphCallRequest is one graph.call runtime step.
*/
type GraphCallRequest struct {
	GraphName string
	Graph     *ast.Graph
	Compute   any
	Inputs    map[string]any
}

/*
GraphCallResult holds graph.call outputs.
*/
type GraphCallResult struct {
	Outputs map[string]any
}

/*
Executor runs a manifest program against a backend and host operations.
*/
type Executor struct {
	backend       Backend
	host          HostOps
	state         *StateStore
	schedulers    map[string]*FlowMatchEulerDiscrete
	stdin         io.Reader
	initialValues map[string]any
}

/*
ExecutorOptions configures program execution.
*/
type ExecutorOptions struct {
	Backend       Backend
	Host          HostOps
	State         *StateStore
	Schedulers    map[string]*FlowMatchEulerDiscrete
	Stdin         io.Reader
	InitialValues map[string]any
}

/*
NewExecutor constructs an Executor.
*/
func NewExecutor(options ExecutorOptions) *Executor {
	return &Executor{
		backend:       options.Backend,
		host:          options.Host,
		state:         options.State,
		schedulers:    options.Schedulers,
		stdin:         options.Stdin,
		initialValues: options.InitialValues,
	}
}

/*
Run executes program steps sequentially.
*/
func (executor *Executor) Run(
	ctx context.Context,
	program *ast.Program,
	graphs map[string]*ast.Graph,
	compute map[string]any,
) error {
	if program == nil {
		return fmt.Errorf("runtime execute: program is required")
	}

	values := make(map[string]any)

	for key, value := range executor.initialValues {
		values[key] = value
	}

	for _, step := range program.Steps {
		if err := executor.runStep(ctx, step, graphs, compute, values); err != nil {
			return fmt.Errorf("runtime step %q op %q: %w", step.ID, step.Op, err)
		}
	}

	return nil
}

func (executor *Executor) runStep(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	if len(step.Body) > 0 {
		return executor.runStepWithBody(ctx, step, graphs, compute, values)
	}

	switch step.Op {
	case "graph.call":
		return executor.runGraphCall(ctx, step, graphs, compute, values)
	case "io.read_line":
		return executor.runReadLine(ctx, step, values)
	case "io.emit_token":
		return executor.runEmitToken(ctx, step, values)
	case "io.write_image":
		return executor.runWriteImage(ctx, step, values)
	case "tokenizer.encode":
		return executor.runEncode(ctx, step, values)
	case "sampling.topk_sample":
		return executor.runTopKSample(ctx, step, values)
	case "value.assign":
		return executor.runAssign(step, values)
	case "scheduler.timesteps":
		return executor.runSchedulerTimesteps(ctx, step, values)
	case "scheduler.step":
		return executor.runSchedulerStep(ctx, step, values)
	case "state.update":
		return executor.runStateUpdate(ctx, step)
	case "control.loop_each":
		return executor.runLoopEach(ctx, step, graphs, compute, values)
	case "control.loop_count":
		return executor.runLoopCount(ctx, step, graphs, compute, values)
	case "control.loop_until_eof":
		return executor.runLoopUntilEOF(ctx, step, graphs, compute, values)
	default:
		return fmt.Errorf("unsupported runtime op %q", step.Op)
	}
}

func (executor *Executor) runStepWithBody(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	switch step.Op {
	case "control.loop_count", "":
		if step.Loop != nil && step.Loop.Repeat != "" {
			return executor.runLoopCount(ctx, step, graphs, compute, values)
		}
	case "control.loop_until_eof":
		return executor.runLoopUntilEOF(ctx, step, graphs, compute, values)
	}

	for _, child := range step.Body {
		if err := executor.runStep(ctx, child, graphs, compute, values); err != nil {
			return err
		}
	}

	return nil
}

func (executor *Executor) runReadLine(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	var text string
	var err error

	if executor.host != nil {
		text, err = executor.host.ReadLine(ctx)
	} else if executor.stdin != nil {
		buffer := make([]byte, 0, 4096)
		chunk := make([]byte, 256)

		for {
			count, readErr := executor.stdin.Read(chunk)

			if count > 0 {
				buffer = append(buffer, chunk[:count]...)
			}

			if readErr == io.EOF {
				break
			}

			if readErr != nil {
				return readErr
			}
		}

		text = strings.TrimSpace(string(buffer))
	} else {
		return fmt.Errorf("io.read_line: no host or stdin configured")
	}

	if err != nil {
		return err
	}

	for _, ref := range step.Out {
		values[ref] = text
	}

	return nil
}

func (executor *Executor) runEncode(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	textRef, ok := step.In["text"]

	if !ok {
		textRef = step.In["value"]
	}

	text, ok := values[textRef].(string)

	if !ok {
		return fmt.Errorf("tokenizer.encode: text %q is not a string", textRef)
	}

	tokenizerName, _ := step.Config["tokenizer"].(string)

	var tokenIDs []int
	var err error

	if executor.host != nil {
		tokenIDs, err = executor.host.Encode(ctx, EncodeRequest{
			Tokenizer: tokenizerName,
			Text:      text,
		})
	} else {
		return fmt.Errorf("tokenizer.encode: host ops are required")
	}

	if err != nil {
		return err
	}

	for _, ref := range step.Out {
		values[ref] = tokenIDs
	}

	return nil
}

func (executor *Executor) runEmitToken(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	for _, ref := range step.In {
		tokenID, err := tokenIDFromValue(values[ref])

		if err != nil {
			return err
		}

		if executor.host == nil {
			return fmt.Errorf("io.emit_token: host ops are required")
		}

		if err := executor.host.EmitToken(ctx, tokenID); err != nil {
			return err
		}
	}

	return nil
}

func (executor *Executor) runWriteImage(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	if executor.host == nil {
		return fmt.Errorf("io.write_image: host ops are required")
	}

	imageRef, ok := step.In["image"]

	if !ok {
		return fmt.Errorf("io.write_image: image input is required")
	}

	path, _ := step.Config["path"].(string)

	return executor.host.WriteImage(ctx, WriteImageRequest{
		Path:   path,
		Tensor: values[imageRef],
	})
}

func (executor *Executor) runTopKSample(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	logitsRef, ok := step.In["value"]

	if !ok {
		for _, ref := range step.In {
			logitsRef = ref
			break
		}
	}

	logits, err := float32Vector(values[logitsRef])

	if err != nil {
		return err
	}

	temperature := float32FromConfig(step.Config, "temperature", 1.0)
	topK := intFromConfig(step.Config, "top_k", 50)

	tokenID := sampleTopK(logits, temperature, topK)

	for _, ref := range step.Out {
		values[ref] = tokenID
	}

	return nil
}

func (executor *Executor) runAssign(step ast.Step, values map[string]any) error {
	for _, ref := range step.In {
		value := values[ref]

		for _, outRef := range step.Out {
			values[outRef] = value
		}
	}

	return nil
}

func (executor *Executor) runSchedulerTimesteps(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	schedulerName, _ := step.Config["scheduler"].(string)

	scheduler, err := executor.scheduler(schedulerName)

	if err != nil {
		return err
	}

	timesteps := scheduler.Timesteps()

	for _, ref := range step.Out {
		values[ref] = timesteps
	}

	return nil
}

func (executor *Executor) runSchedulerStep(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	schedulerName, _ := step.Config["scheduler"].(string)

	scheduler, err := executor.scheduler(schedulerName)

	if err != nil {
		return err
	}

	latents, err := executor.resolveTensorInput(step.In["latents"], values)

	if err != nil {
		return err
	}

	velocity, err := float32Vector(values[step.In["velocity"]])

	if err != nil {
		return err
	}

	timestep := float32(0)

	if executor.state != nil {
		stepIndex, ok := executor.state.Get("step_index")

		if ok {
			if counter, ok := stepIndex.(int64); ok {
				timesteps, ok := values["timesteps"].([]float32)

				if ok && int(counter) < len(timesteps) {
					timestep = timesteps[counter]
				}
			}
		}
	}

	updated, err := scheduler.Step(latents, velocity, timestep)

	if err != nil {
		return err
	}

	if executor.state != nil && step.In["latents"] != "" {
		if strings.HasPrefix(step.In["latents"], "state.") {
			name := step.In["latents"][len("state."):]
			executor.state.Set(name, updated)
		}
	}

	for _, ref := range step.Out {
		if strings.HasPrefix(ref, "state.") {
			name := ref[len("state."):]
			executor.state.Set(name, updated)
			continue
		}

		values[ref] = updated
	}

	return nil
}

func (executor *Executor) runStateUpdate(ctx context.Context, step ast.Step) error {
	_ = ctx

	if executor.state == nil {
		return fmt.Errorf("state.update: state store is required")
	}

	update, _ := step.Config["update"].(string)
	target, _ := step.Config["target"].(string)

	return executor.state.Update(update, target)
}

func (executor *Executor) runLoopEach(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	sourceName := step.Loop.Over

	if sourceName == "" {
		sourceName = step.In["source"]
	}

	source, ok := values[sourceName]

	if !ok {
		return fmt.Errorf("control.loop_each: unknown source %q", sourceName)
	}

	timesteps, ok := source.([]float32)

	if !ok {
		return fmt.Errorf("control.loop_each: source %q must be []float32 timesteps", sourceName)
	}

	itemName := step.Loop.As

	if itemName == "" {
		itemName = "timestep"
	}

	for _, timestep := range timesteps {
		values[itemName] = timestep

		for _, child := range step.Body {
			if err := executor.runStep(ctx, child, graphs, compute, values); err != nil {
				return err
			}
		}
	}

	return nil
}

func (executor *Executor) runLoopCount(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	repeatText := ""

	if step.Loop != nil {
		repeatText = step.Loop.Repeat
	}

	count, err := parseRepeatCount(repeatText)

	if err != nil {
		return err
	}

	for iteration := 0; iteration < count; iteration++ {
		values["__loop_index__"] = iteration

		for _, child := range step.Body {
			if err := executor.runStep(ctx, child, graphs, compute, values); err != nil {
				return err
			}
		}
	}

	return nil
}

func (executor *Executor) runLoopUntilEOF(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	for {
		beforeLen := len(values)

		for _, child := range step.Body {
			if err := executor.runStep(ctx, child, graphs, compute, values); err != nil {
				if err == io.EOF {
					return nil
				}

				return err
			}
		}

		if len(values) == beforeLen {
			return nil
		}

		line, ok := values["user_text"].(string)

		if ok && line == "" {
			return nil
		}
	}
}

func (executor *Executor) runGraphCall(
	ctx context.Context,
	step ast.Step,
	graphs map[string]*ast.Graph,
	compute map[string]any,
	values map[string]any,
) error {
	graphName := step.Graph

	if graphName == "" {
		configured, ok := step.Config["graph"].(string)

		if !ok || configured == "" {
			return fmt.Errorf("graph.call requires graph name")
		}

		graphName = configured
	}

	graph, ok := graphs[graphName]

	if !ok {
		return fmt.Errorf("unknown graph %q", graphName)
	}

	inputs := make(map[string]any, len(step.In))

	for name, ref := range step.In {
		if strings.HasPrefix(ref, "state.") && executor.state != nil {
			value, err := executor.state.ResolveReference(ref)

			if err != nil {
				return err
			}

			inputs[name] = value
			continue
		}

		inputs[name] = values[ref]
	}

	result, err := executor.backend.CallGraph(ctx, GraphCallRequest{
		GraphName: graphName,
		Graph:     graph,
		Compute:   compute[graphName],
		Inputs:    inputs,
	})

	if err != nil {
		return err
	}

	for name, ref := range step.Out {
		if value, ok := result.Outputs[name]; ok {
			values[ref] = value
		}
	}

	return nil
}

func (executor *Executor) scheduler(name string) (*FlowMatchEulerDiscrete, error) {
	if executor.schedulers == nil {
		return nil, fmt.Errorf("scheduler %q: no schedulers configured", name)
	}

	scheduler, ok := executor.schedulers[name]

	if !ok {
		return nil, fmt.Errorf("scheduler %q: not found", name)
	}

	return scheduler, nil
}

func (executor *Executor) resolveTensorInput(
	reference string,
	values map[string]any,
) ([]float32, error) {
	if strings.HasPrefix(reference, "state.") && executor.state != nil {
		value, err := executor.state.ResolveReference(reference)

		if err != nil {
			return nil, err
		}

		return float32Vector(value)
	}

	return float32Vector(values[reference])
}

func sampleTopK(logits []float32, temperature float32, topK int) int {
	if len(logits) == 0 {
		return 0
	}

	if topK <= 0 || topK > len(logits) {
		topK = len(logits)
	}

	type scored struct {
		index int
		value float32
	}

	scores := make([]scored, len(logits))

	for index, value := range logits {
		scores[index] = scored{index: index, value: value}
	}

	sort.Slice(scores, func(left, right int) bool {
		return scores[left].value > scores[right].value
	})

	candidates := scores[:topK]
	maxLogit := candidates[0].value
	weights := make([]float64, len(candidates))
	sum := 0.0

	for index, candidate := range candidates {
		scaled := float64((candidate.value - maxLogit) / temperature)
		weights[index] = mathExp(scaled)
		sum += weights[index]
	}

	threshold := rand.Float64() * sum
	acc := 0.0

	for index, weight := range weights {
		acc += weight

		if acc >= threshold {
			return candidates[index].index
		}
	}

	return candidates[len(candidates)-1].index
}

func mathExp(value float64) float64 {
	if value < -64 {
		return 0
	}

	result := 1.0
	term := 1.0

	for termIndex := 1; termIndex <= 12; termIndex++ {
		term *= value / float64(termIndex)
		result += term
	}

	return result
}

func parseRepeatCount(text string) (int, error) {
	text = strings.TrimSpace(text)

	if text == "" {
		return 0, fmt.Errorf("loop repeat count is required")
	}

	var count int

	if _, err := fmt.Sscanf(text, "%d", &count); err != nil {
		return 0, fmt.Errorf("loop repeat count %q: %w", text, err)
	}

	return count, nil
}

func tokenIDFromValue(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	default:
		return 0, fmt.Errorf("token value has unsupported type %T", value)
	}
}

func float32Vector(value any) ([]float32, error) {
	switch typed := value.(type) {
	case []float32:
		return typed, nil
	case []float64:
		out := make([]float32, len(typed))

		for index, element := range typed {
			out[index] = float32(element)
		}

		return out, nil
	default:
		return nil, fmt.Errorf("expected float32 vector, got %T", value)
	}
}

func float32FromConfig(config map[string]any, key string, fallback float32) float32 {
	raw, ok := config[key]

	if !ok {
		return fallback
	}

	switch typed := raw.(type) {
	case float64:
		return float32(typed)
	case float32:
		return typed
	case int:
		return float32(typed)
	default:
		return fallback
	}
}

func intFromConfig(config map[string]any, key string, fallback int) int {
	raw, ok := config[key]

	if !ok {
		return fallback
	}

	switch typed := raw.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return fallback
	}
}
