package runtime

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/tensor"
)

/*
StateStore holds runtime state objects declared by a program manifest.
*/
type StateStore struct {
	mu           sync.Mutex
	slots        map[string]any
	declarations []ast.StateDeclaration
}

/*
PagedTensorState describes generic resident paged tensor storage.
The runtime treats it as an opaque handle; manifests decide how pages are used.
*/
type PagedTensorState struct {
	Shape     []int
	PageSize  int
	PageCount int
	Storage   any
}

/*
PageTableState stores generic page indices for a paged state object.
*/
type PageTableState struct {
	Capacity int
	Pages    []int32
	Storage  any
}

/*
NewStateStore constructs state from program declarations.
*/
func NewStateStore(declarations []ast.StateDeclaration) (*StateStore, error) {
	store := &StateStore{
		slots:        make(map[string]any, len(declarations)),
		declarations: declarations,
	}

	for _, declaration := range declarations {
		value, err := store.initialize(declaration)

		if err != nil {
			return nil, err
		}

		store.slots[declaration.Name] = value
	}

	return store, nil
}

func (store *StateStore) initialize(declaration ast.StateDeclaration) (any, error) {
	switch declaration.Type {
	case "counter":
		return int64(0), nil
	case "tensor":
		return store.initializeTensor(declaration)
	case "paged_tensor":
		return store.initializePagedTensor(declaration)
	case "page_table":
		return store.initializePageTable(declaration)
	default:
		return nil, fmt.Errorf("state %q: unsupported type %q", declaration.Name, declaration.Type)
	}
}

func (store *StateStore) initializePagedTensor(declaration ast.StateDeclaration) (any, error) {
	shape, err := intsFromAnySlice(declaration.Shape)

	if err != nil {
		return nil, fmt.Errorf("state %q: paged tensor shape: %w", declaration.Name, err)
	}

	return &PagedTensorState{
		Shape:     shape,
		PageSize:  intFromMap(declaration.Config, "page_size", 0),
		PageCount: intFromMap(declaration.Config, "page_count", 0),
	}, nil
}

func (store *StateStore) initializePageTable(declaration ast.StateDeclaration) (any, error) {
	return &PageTableState{
		Capacity: intFromMap(declaration.Config, "capacity", 0),
		Pages:    []int32{},
	}, nil
}

func (store *StateStore) initializeTensor(declaration ast.StateDeclaration) (any, error) {
	switch declaration.Init {
	case "gaussian":
		seed := int64FromAny(declaration.Seed, 0)
		rng := rand.New(rand.NewSource(seed))
		elementCount := stateElementCount(declaration.Shape)

		values := make([]float32, elementCount)

		for index := range values {
			values[index] = float32(rng.NormFloat64())
		}

		return values, nil
	default:
		return make([]float32, stateElementCount(declaration.Shape)), nil
	}
}

func stateElementCount(shape []any) int64 {
	elementCount := int64(1)

	for _, dimension := range shape {
		switch typed := dimension.(type) {
		case int:
			elementCount *= int64(typed)
		case int64:
			elementCount *= typed
		case float64:
			elementCount *= int64(typed)
		}
	}

	return elementCount
}

func int64FromAny(value any, fallback int64) int64 {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int64:
		return typed
	case float64:
		return int64(typed)
	default:
		return fallback
	}
}

/*
Get returns one state slot by name.
*/
func (store *StateStore) Get(name string) (any, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()

	value, ok := store.slots[name]

	return value, ok
}

/*
Set replaces one state slot.
*/
func (store *StateStore) Set(name string, value any) {
	store.mu.Lock()
	defer store.mu.Unlock()

	store.slots[name] = value
}

/*
SetReference writes state.cache style references.
*/
func (store *StateStore) SetReference(reference string, value any) error {
	if !strings.HasPrefix(reference, "state.") {
		return fmt.Errorf("state reference %q must start with state.", reference)
	}

	name := reference[len("state."):]

	store.mu.Lock()
	defer store.mu.Unlock()

	previous := store.slots[name]
	store.slots[name] = value

	closeReplacedStateValue(previous, value)

	return nil
}

/*
ResolveReference reads state.latents style references.
*/
func (store *StateStore) ResolveReference(reference string) (any, error) {
	if reference == "" {
		return nil, fmt.Errorf("state reference is required")
	}

	if !strings.HasPrefix(reference, "state.") {
		return nil, fmt.Errorf("state reference %q must start with state.", reference)
	}

	name := reference[len("state."):]

	value, ok := store.Get(name)

	if !ok {
		return nil, fmt.Errorf("unknown state %q", name)
	}

	return value, nil
}

func closeReplacedStateValue(previous any, next any) {
	previousTensor, previousIsTensor := previous.(tensor.Tensor)
	nextTensor, nextIsTensor := next.(tensor.Tensor)

	if previousIsTensor && (!nextIsTensor || previousTensor != nextTensor) {
		_ = previousTensor.Close()
	}
}

func intsFromAnySlice(values []any) ([]int, error) {
	out := make([]int, 0, len(values))

	for _, value := range values {
		switch typed := value.(type) {
		case int:
			out = append(out, typed)
		case int64:
			out = append(out, int(typed))
		case float64:
			out = append(out, int(typed))
		default:
			return nil, fmt.Errorf("unsupported dimension %T", value)
		}
	}

	return out, nil
}

func intFromMap(values map[string]any, key string, fallback int) int {
	if values == nil {
		return fallback
	}

	value, ok := values[key]

	if !ok {
		return fallback
	}

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

/*
Update applies state.update operations.
*/
func (store *StateStore) Update(update string, target string) error {
	if !strings.HasPrefix(target, "state.") {
		return fmt.Errorf("state update target %q must start with state.", target)
	}

	name := target[len("state."):]

	store.mu.Lock()
	defer store.mu.Unlock()

	value, ok := store.slots[name]

	if !ok {
		return fmt.Errorf("unknown state %q", name)
	}

	switch update {
	case "increment":
		counter, ok := value.(int64)

		if !ok {
			return fmt.Errorf("state %q is not a counter", name)
		}

		store.slots[name] = counter + 1

		return nil
	default:
		return fmt.Errorf("unsupported state update %q", update)
	}
}

/*
LatentNorm returns the L2 norm of a float32 latent buffer.
*/
func LatentNorm(values []float32) float64 {
	sum := 0.0

	for _, value := range values {
		sum += float64(value) * float64(value)
	}

	return math.Sqrt(sum)
}
