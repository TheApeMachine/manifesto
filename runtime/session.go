package runtime

import (
	"context"
	"fmt"
	"io"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

/*
ProgramSession owns program-runtime state for an already compiled program.
*/
type ProgramSession struct {
	program     *ast.Program
	graphs      map[string]*ast.Graph
	compute     map[string]*ir.Graph
	plans       map[string]*ExecutionPlan
	backend     Backend
	host        HostOps
	state       *StateStore
	stateMemory tensor.Backend
	schedulers  map[string]*FlowMatchEulerDiscrete
	stdin       io.Reader
}

/*
ProgramSessionOptions wires host-provided dependencies into manifesto runtime.
*/
type ProgramSessionOptions struct {
	Program      *ast.Program
	Graphs       map[string]*ast.Graph
	Compute      map[string]*ir.Graph
	Backend      Backend
	Host         HostOps
	State        *StateStore
	Schedulers   map[string]*FlowMatchEulerDiscrete
	Stdin        io.Reader
	StateBackend tensor.Backend
}

/*
NewProgramSession constructs runtime state for an already compiled program.
*/
func NewProgramSession(options ProgramSessionOptions) (*ProgramSession, error) {
	if options.Program == nil {
		return nil, fmt.Errorf("runtime session: program is required")
	}

	state := options.State
	var err error

	if state == nil {
		state, err = NewStateStore(options.Program.State)

		if err != nil {
			return nil, err
		}
	}

	if options.StateBackend != nil {
		if err := MaterializeStateTensors(
			state,
			options.Program.State,
			options.StateBackend,
			RuntimeExecutionDType(options.Graphs),
		); err != nil {
			return nil, err
		}
	}

	schedulers := options.Schedulers

	if schedulers == nil {
		schedulers, err = SchedulersFromProgram(options.Program)

		if err != nil {
			return nil, err
		}
	}

	plans, err := ExecutionPlansFromCompute(options.Compute)

	if err != nil {
		return nil, err
	}

	return &ProgramSession{
		program:     options.Program,
		graphs:      options.Graphs,
		compute:     options.Compute,
		plans:       plans,
		backend:     options.Backend,
		host:        options.Host,
		state:       state,
		stateMemory: options.StateBackend,
		schedulers:  schedulers,
		stdin:       options.Stdin,
	}, nil
}

/*
Run executes the compiled program.
*/
func (session *ProgramSession) Run(ctx context.Context) error {
	return session.RunWithValues(ctx, nil)
}

/*
RunWithValues executes the compiled program with pre-populated values.
*/
func (session *ProgramSession) RunWithValues(ctx context.Context, initial map[string]any) error {
	if session == nil {
		return fmt.Errorf("runtime session: session is required")
	}

	executor := NewExecutor(ExecutorOptions{
		Backend:       session.backend,
		Host:          session.host,
		State:         session.state,
		StateMemory:   session.stateMemory,
		Schedulers:    session.schedulers,
		Plans:         session.plans,
		Stdin:         session.stdin,
		InitialValues: initial,
	})

	computeAny := make(map[string]any, len(session.compute))

	for name, graph := range session.compute {
		computeAny[name] = graph
	}

	return executor.Run(ctx, session.program, session.graphs, computeAny)
}

/*
ExecutionPlansFromCompute builds cached execution plans for every compute graph.
*/
func ExecutionPlansFromCompute(compute map[string]*ir.Graph) (map[string]*ExecutionPlan, error) {
	plans := make(map[string]*ExecutionPlan, len(compute))

	for graphName, computeGraph := range compute {
		plan, err := NewExecutionPlan(graphName, computeGraph)

		if err != nil {
			return nil, err
		}

		plans[graphName] = plan
	}

	return plans, nil
}

/*
SchedulersFromProgram constructs runtime schedulers from program declarations.
*/
func SchedulersFromProgram(program *ast.Program) (map[string]*FlowMatchEulerDiscrete, error) {
	schedulers := make(map[string]*FlowMatchEulerDiscrete)

	if program == nil {
		return schedulers, nil
	}

	for name, declaration := range program.Schedulers {
		switch declaration.Type {
		case "flow_match_euler_discrete":
			scheduler, err := NewFlowMatchEulerDiscrete(SchedulerConfig{
				Steps:             intFromAny(declaration.Config["steps"], 28),
				NumTrainTimesteps: intFromAny(declaration.Config["num_train_timesteps"], 1000),
				Shift:             float64FromAny(declaration.Config["shift"], 1),
				UseDynamicShift:   boolFromAny(declaration.Config["use_dynamic_shifting"]),
				TimeShiftType:     stringFromAny(declaration.Config["time_shift_type"], "exponential"),
				ImageSeqLen:       intFromAny(declaration.Config["image_seq_len"], 4096),
			})

			if err != nil {
				return nil, err
			}

			schedulers[name] = scheduler
		default:
			return nil, fmt.Errorf("runtime session: unsupported scheduler type %q", declaration.Type)
		}
	}

	return schedulers, nil
}

/*
MaterializeStateTensors stores tensor state as resident tensors on memory.
*/
func MaterializeStateTensors(
	stateStore *StateStore,
	declarations []ast.StateDeclaration,
	memory tensor.Backend,
	storageDType dtype.DType,
) error {
	if stateStore == nil {
		return fmt.Errorf("runtime session: state store is required")
	}

	if memory == nil {
		return fmt.Errorf("runtime session: tensor backend is required")
	}

	if !storageDType.IsFloat() {
		storageDType = dtype.Float32
	}

	for _, declaration := range declarations {
		if declaration.Type != "tensor" {
			continue
		}

		if err := materializeStateTensor(stateStore, declaration, memory, storageDType); err != nil {
			return err
		}
	}

	return nil
}

func materializeStateTensor(
	stateStore *StateStore,
	declaration ast.StateDeclaration,
	memory tensor.Backend,
	storageDType dtype.DType,
) error {
	value, ok := stateStore.Get(declaration.Name)

	if !ok {
		return nil
	}

	if _, ok := value.(tensor.Tensor); ok {
		return nil
	}

	values, ok := value.([]float32)

	if !ok {
		return fmt.Errorf("runtime session: state %q is %T, expected []float32", declaration.Name, value)
	}

	shape, err := StateTensorShape(declaration)

	if err != nil {
		return err
	}

	tensorValue, err := memory.Upload(shape, storageDType, Float32AsDTypeBytes(values, storageDType))

	if err != nil {
		return err
	}

	stateStore.Set(declaration.Name, tensorValue)

	return nil
}

/*
RuntimeExecutionDType returns the floating execution dtype for runtime state.
*/
func RuntimeExecutionDType(graphs map[string]*ast.Graph) dtype.DType {
	for _, graph := range graphs {
		if graph == nil || !graph.ExecutionDType.IsFloat() {
			continue
		}

		return graph.ExecutionDType
	}

	return dtype.Float32
}

/*
StateTensorShape resolves a state declaration shape into a tensor shape.
*/
func StateTensorShape(declaration ast.StateDeclaration) (tensor.Shape, error) {
	dims := make([]int, len(declaration.Shape))

	for index, dimension := range declaration.Shape {
		switch typed := dimension.(type) {
		case int:
			dims[index] = typed
		case int64:
			dims[index] = int(typed)
		case float64:
			dims[index] = int(typed)
		default:
			return tensor.Shape{}, fmt.Errorf("runtime session: unsupported state dimension %T", dimension)
		}
	}

	return tensor.NewShape(dims)
}

/*
Float32AsDTypeBytes encodes float32 values into the requested float dtype.
*/
func Float32AsDTypeBytes(values []float32, targetDType dtype.DType) []byte {
	switch targetDType {
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded)
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for index, value := range values {
			encoded[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded)
	default:
		return convert.Float32ToBytes(values)
	}
}

func float64FromAny(value any, fallback float64) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return fallback
	}
}

func boolFromAny(value any) bool {
	typed, ok := value.(bool)

	if !ok {
		return false
	}

	return typed
}

func stringFromAny(value any, fallback string) string {
	typed, ok := value.(string)

	if !ok {
		return fallback
	}

	return typed
}

func intFromAny(value any, fallback int) int {
	switch typed := value.(type) {
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
