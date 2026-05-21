package runtime

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"

	"github.com/theapemachine/manifesto/ast"
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
	default:
		return nil, fmt.Errorf("state %q: unsupported type %q", declaration.Name, declaration.Type)
	}
}

func (store *StateStore) initializeTensor(declaration ast.StateDeclaration) (any, error) {
	switch declaration.Init {
	case "gaussian":
		seed := int64(0)

		if declaration.Seed != nil {
			switch typed := declaration.Seed.(type) {
			case int:
				seed = int64(typed)
			case int64:
				seed = typed
			case float64:
				seed = int64(typed)
			}
		}

		rng := rand.New(rand.NewSource(seed))
		elementCount := int64(1)

		for _, dimension := range declaration.Shape {
			switch typed := dimension.(type) {
			case int:
				elementCount *= int64(typed)
			case int64:
				elementCount *= typed
			case float64:
				elementCount *= int64(typed)
			}
		}

		values := make([]float32, elementCount)

		for index := range values {
			values[index] = float32(rng.NormFloat64())
		}

		return values, nil
	default:
		return make([]float32, 0), nil
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
