package runtime

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
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
WorkspaceAttacher is the optional capability a Backend advertises when
it can consume the planner's WorkspaceLayout. The orchestrator type-
asserts on this after CompileAssets and, for every compiled graph that
the planner returned a topology for, calls AttachWorkspace so the
backend can pre-allocate and pre-resolve per-node tensors before the
first dispatch.

Backends without workspace support (test mocks, future XLA bridges
that own their own residency) simply don't implement this interface
and the orchestrator skips the attach step. Backends that do (the
puter execution.Backend) get the planner output handed in here.
*/
type WorkspaceAttacher interface {
	AttachWorkspace(
		graphName string,
		graph *ast.Graph,
		topology *ir.Topology,
	) error
}

/*
GraphCallRequest is one graph.call runtime step.
*/
type GraphCallRequest struct {
	GraphName    string
	Graph        *ast.Graph
	Compute      any
	Plan         *ExecutionPlan
	Inputs       map[string]any
	StateOutputs map[string]bool
	// LaunchBindings carries per-invocation SymbolMap values (live
	// sequence length, batch size, …) derived from concrete inputs.
	// The execution backend substitutes these for planner upper bounds
	// when resolving bind shapes and shape intrinsics.
	LaunchBindings ir.SymbolMap
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
	backend        Backend
	host           HostOps
	state          *StateStore
	stateMemory    tensor.Backend
	executionDType dtype.DType
	plans          map[string]*ExecutionPlan
	stdin          io.Reader
	initialValues  map[string]any
}

/*
ExecutorOptions configures program execution.
*/
type ExecutorOptions struct {
	Backend        Backend
	Host           HostOps
	State          *StateStore
	StateMemory    tensor.Backend
	ExecutionDType dtype.DType
	Plans          map[string]*ExecutionPlan
	Stdin          io.Reader
	InitialValues  map[string]any
}

/*
NewExecutor constructs an Executor.
*/
func NewExecutor(options ExecutorOptions) *Executor {
	return &Executor{
		backend:        options.Backend,
		host:           options.Host,
		state:          options.State,
		stateMemory:    options.StateMemory,
		executionDType: options.ExecutionDType,
		plans:          options.Plans,
		stdin:          options.Stdin,
		initialValues:  options.InitialValues,
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

	defer executor.closeRuntimeValues(values)

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
	case "value.append":
		return executor.runAppend(step, values)
	case "math.axpy":
		return executor.runAxpy(ctx, step, values)
	case "math.linspace":
		return executor.runLinspace(step, values)
	case "math.scalar_broadcast":
		return executor.runScalarBroadcast(ctx, step, values)
	case "random.normal":
		return executor.runRandomNormal(step, values)
	case "state.update":
		return executor.runStateUpdate(ctx, step)
	case "state.paged_plan":
		return executor.runPagedPlan(ctx, step, values)
	case "state.advance_position":
		return executor.runAdvancePosition(ctx, step, values)
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
	case "control.loop_each":
		return executor.runLoopEach(ctx, step, graphs, compute, values)
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
	tokenizerFile, _ := step.Config["tokenizer_file"].(string)
	applyChatTemplate, _ := step.Config["apply_chat_template"].(bool)
	appendTo, _ := step.Config["append_to"].(string)
	maxLength := intFromConfig(step.Config, "max_length", 0)
	padTokenID := intFromConfig(step.Config, "pad_token_id", 0)
	chatContinuation := false

	if appendTo != "" {
		existing, ok := values[appendTo].([]int)
		chatContinuation = ok && len(existing) > 0
	}

	var tokenIDs []int
	var err error

	if executor.host != nil {
		tokenIDs, err = executor.host.Encode(ctx, EncodeRequest{
			Tokenizer:         tokenizerName,
			TokenizerFile:     tokenizerFile,
			Text:              text,
			ApplyChatTemplate: applyChatTemplate,
			ChatContinuation:  chatContinuation,
			MaxLength:         maxLength,
			PadTokenID:        padTokenID,
		})
	} else {
		return fmt.Errorf("tokenizer.encode: host ops are required")
	}

	if err != nil {
		return err
	}

	if appendTo != "" {
		existing, ok := values[appendTo].([]int)

		if ok {
			tokenIDs = append(append([]int(nil), existing...), tokenIDs...)
		}
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

		tokenizerName, _ := step.Config["tokenizer"].(string)
		tokenizerFile, _ := step.Config["tokenizer_file"].(string)
		if err := executor.host.EmitToken(ctx, EmitTokenRequest{
			Tokenizer:     tokenizerName,
			TokenizerFile: tokenizerFile,
			TokenID:       tokenID,
		}); err != nil {
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
	layout, _ := step.Config["layout"].(string)
	valueRange, _ := step.Config["range"].(string)

	return executor.host.WriteImage(ctx, WriteImageRequest{
		Path:     path,
		Tensor:   values[imageRef],
		Width:    intFromConfig(step.Config, "width", 0),
		Height:   intFromConfig(step.Config, "height", 0),
		Channels: intFromConfig(step.Config, "channels", 0),
		Layout:   layout,
		Range:    valueRange,
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

	logitsValue, ok := values[logitsRef]

	if !ok {
		return fmt.Errorf("sampling.topk_sample: unknown logits value %q", logitsRef)
	}

	logits, err := float32Vector(ctx, logitsValue)

	if err != nil {
		return err
	}

	vocabSize := intFromConfig(step.Config, "vocab_size", 0)
	if vocabSize > 0 && len(logits) > vocabSize {
		// Only sample from the last token's logits
		logits = logits[len(logits)-vocabSize:]
	}

	temperature := float32FromConfig(step.Config, "temperature", 1.0)
	topK := intFromConfig(step.Config, "top_k", 50)

	tokenID, err := sampleTopK(logits, temperature, topK)

	if err != nil {
		return err
	}

	stopTokenIDs := intSliceFromConfig(step.Config, "stop_token_ids")

	for _, stopTokenID := range stopTokenIDs {
		if tokenID == stopTokenID {
			values["__loop_break__"] = true
			break
		}
	}

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

func (executor *Executor) runAxpy(
	ctx context.Context,
	step ast.Step,
	values map[string]any,
) error {
	_ = ctx

	yRef, ok := step.In["y"]

	if !ok {
		return fmt.Errorf("math.axpy: y input is required")
	}

	xRef, ok := step.In["x"]

	if !ok {
		return fmt.Errorf("math.axpy: x input is required")
	}

	alphaRef, ok := step.In["alpha"]

	if !ok {
		return fmt.Errorf("math.axpy: alpha input is required")
	}

	yValue, err := executor.resolveValue(yRef, values)

	if err != nil {
		return err
	}

	xValue, err := executor.resolveValue(xRef, values)

	if err != nil {
		return err
	}

	alphaValue, err := executor.resolveValue(alphaRef, values)

	if err != nil {
		return err
	}

	addend, err := float32Vector(ctx, xValue)

	if err != nil {
		return err
	}

	alpha := float32FromAny(alphaValue, 0)

	updated, err := axpyOnto(executor.stateMemory, yValue, addend, alpha)

	if err != nil {
		return err
	}

	for _, ref := range step.Out {
		if strings.HasPrefix(ref, "state.") && executor.state != nil {
			if err := executor.state.SetReference(ref, updated); err != nil {
				return err
			}

			continue
		}

		setRuntimeValue(values, ref, updated)
	}

	return nil
}

func (executor *Executor) runAppend(step ast.Step, values map[string]any) error {
	targetRef, ok := step.Config["target"].(string)
	if !ok {
		return fmt.Errorf("value.append requires a target config")
	}

	targetValue, ok := values[targetRef]
	if !ok {
		return fmt.Errorf("value.append target %q not found", targetRef)
	}

	for _, ref := range step.In {
		value := values[ref]

		switch typedTarget := targetValue.(type) {
		case []int:
			if tokenID, err := tokenIDFromValue(value); err == nil {
				targetValue = append(typedTarget, tokenID)
			} else {
				return fmt.Errorf("value.append cannot append %T to []int", value)
			}
		default:
			return fmt.Errorf("value.append unsupported target type %T", targetValue)
		}
	}

	values[targetRef] = targetValue

	for _, outRef := range step.Out {
		values[outRef] = targetValue
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

	delete(values, "__loop_break__")

	for iteration := 0; iteration < count; iteration++ {
		values["__loop_index__"] = iteration

		for _, child := range step.Body {
			if err := executor.runStep(ctx, child, graphs, compute, values); err != nil {
				return err
			}
		}

		shouldBreak, ok := values["__loop_break__"].(bool)

		if ok && shouldBreak {
			delete(values, "__loop_break__")
			return nil
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

			if child.Op == "io.read_line" {
				line, ok := values["user_text"].(string)

				if ok && line == "" {
					return nil
				}
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

	if executor.backend == nil {
		return fmt.Errorf("graph.call %q requires a backend", graphName)
	}

	inputs := make(map[string]any, len(step.In))

	for name, ref := range step.In {
		if strings.HasPrefix(ref, "state.") && executor.state != nil {
			value, err := executor.state.ResolveReference(ref)

			if err != nil {
				return err
			}

			resolved, err := ResolveGraphInputForGraph(graph, name, value, executor.stateMemory)

			if err != nil {
				return fmt.Errorf("graph.call input %q: %w", name, err)
			}

			inputs[name] = resolved
			continue
		}

		rawValue := values[ref]

		resolved, err := ResolveGraphInputForGraph(graph, name, rawValue, executor.stateMemory)

		if err != nil {
			return fmt.Errorf("graph.call input %q: %w", name, err)
		}

		inputs[name] = resolved
	}

	result, err := executor.backend.CallGraph(ctx, GraphCallRequest{
		GraphName:      graphName,
		Graph:          graph,
		Compute:        compute[graphName],
		Plan:           executor.plans[graphName],
		Inputs:         inputs,
		StateOutputs:   stateOutputsForStep(step),
		LaunchBindings: DeriveLaunchBindings(graph, inputs),
	})

	if err != nil {
		return err
	}

	for name, ref := range step.Out {
		if value, ok := result.Outputs[name]; ok {
			if strings.HasPrefix(ref, "state.") && executor.state != nil {
				previous, _ := executor.state.ResolveReference(ref)

				committed, err := CommitGraphStateOutput(ref, previous, value)

				if err != nil {
					return err
				}

				if err := executor.state.SetReference(ref, committed); err != nil {
					return err
				}

				continue
			}

			setRuntimeValue(values, ref, value)
		}
	}

	return nil
}

func stateOutputsForStep(step ast.Step) map[string]bool {
	outputs := make(map[string]bool)

	for name, ref := range step.Out {
		if strings.HasPrefix(ref, "state.") {
			outputs[name] = true
		}
	}

	return outputs
}

func (executor *Executor) resolveValue(reference string, values map[string]any) (any, error) {
	if strings.HasPrefix(reference, "state.") && executor.state != nil {
		return executor.state.ResolveReference(reference)
	}

	value, ok := values[reference]

	if !ok {
		return nil, fmt.Errorf("unknown value %q", reference)
	}

	return value, nil
}

type topKCandidate struct {
	index int
	value float32
}

func sampleTopK(logits []float32, temperature float32, topK int) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("sampling.topk_sample: logits are empty")
	}

	if !(temperature > 0) {
		return 0, fmt.Errorf("sampling.topk_sample: temperature must be positive")
	}

	if topK <= 0 || topK > len(logits) {
		topK = len(logits)
	}

	scores := make([]topKCandidate, len(logits))

	for index, value := range logits {
		scores[index] = topKCandidate{index: index, value: value}
	}

	sort.Slice(scores, func(left, right int) bool {
		return scores[left].value > scores[right].value
	})

	candidates := scores[:topK]
	weights, sum := topKWeights(candidates, temperature)
	threshold := rand.Float64() * sum
	accumulator := 0.0

	for index, weight := range weights {
		accumulator += weight

		if accumulator >= threshold {
			return candidates[index].index, nil
		}
	}

	return candidates[len(candidates)-1].index, nil
}

func topKWeights(candidates []topKCandidate, temperature float32) ([]float64, float64) {
	maxLogit := candidates[0].value
	weights := make([]float64, len(candidates))
	sum := 0.0

	for index, candidate := range candidates {
		scaled := float64((candidate.value - maxLogit) / temperature)
		weights[index] = math.Exp(scaled)
		sum += weights[index]
	}

	return weights, sum
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

func float32Vector(ctx context.Context, value any) ([]float32, error) {
	switch typed := value.(type) {
	case []float32:
		return typed, nil
	case []float64:
		out := make([]float32, len(typed))

		for index, element := range typed {
			out[index] = float32(element)
		}

		return out, nil
	case tensor.Tensor:
		if !typed.DType().IsFloat() {
			return nil, fmt.Errorf("expected float tensor, got %s", typed.DType())
		}

		if err := typed.Sync(ctx); err != nil {
			return nil, err
		}

		if typed.Location() == tensor.Host && typed.DType() == dtype.Float32 {
			return typed.Float32Native()
		}

		if typed.Location() != tensor.Host {
			return float32VectorFromRawTensor(typed)
		}

		return float32VectorFromRawTensor(typed)
	default:
		return nil, fmt.Errorf("expected float32 vector, got %T", value)
	}
}

func float32VectorFromRawTensor(value tensor.Tensor) ([]float32, error) {
	dataType, rawBytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return convert.BytesToFloat32(dataType, rawBytes)
}

func setRuntimeValue(values map[string]any, ref string, value any) {
	if previous, ok := values[ref]; ok {
		previousTensor, previousIsTensor := previous.(tensor.Tensor)
		nextTensor, nextIsTensor := value.(tensor.Tensor)

		if previousIsTensor && (!nextIsTensor || previousTensor != nextTensor) {
			closeRuntimeValue(previous)
		}
	}

	values[ref] = value
}

func (executor *Executor) closeRuntimeValues(values map[string]any) {
	stateTensors := executor.stateTensorSet()

	for ref, value := range values {
		if executor.initialValues != nil {
			if _, ok := executor.initialValues[ref]; ok {
				continue
			}
		}

		tensorValue, ok := value.(tensor.Tensor)

		if ok && tensorValue != nil && stateTensors[tensorValue] {
			continue
		}

		closeRuntimeValue(value)
	}
}

func (executor *Executor) stateTensorSet() map[tensor.Tensor]bool {
	owned := make(map[tensor.Tensor]bool)

	if executor.state == nil {
		return owned
	}

	for _, value := range executor.state.AllSlots() {
		tensorValue, ok := value.(tensor.Tensor)

		if !ok || tensorValue == nil {
			continue
		}

		owned[tensorValue] = true
	}

	return owned
}

func closeRuntimeValue(value any) {
	tensorValue, ok := value.(tensor.Tensor)

	if !ok || tensorValue == nil {
		return
	}

	_ = tensorValue.Close()
}

func float32FromAny(value any, fallback float32) float32 {
	switch typed := value.(type) {
	case float32:
		return typed
	case float64:
		return float32(typed)
	case int:
		return float32(typed)
	case int64:
		return float32(typed)
	default:
		return fallback
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

func intSliceFromConfig(config map[string]any, key string) []int {
	raw, ok := config[key]

	if !ok {
		return nil
	}

	switch typed := raw.(type) {
	case []int:
		return typed
	case []any:
		values := make([]int, 0, len(typed))

		for _, value := range typed {
			values = append(values, intFromAny(value, 0))
		}

		return values
	default:
		return nil
	}
}
