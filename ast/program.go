package ast

/*
Program is a manifest runtime program: IO, control flow, graph calls, and state.
*/
type Program struct {
	Name           string
	Includes       map[string]string
	IncludeObjects map[string]any
	Variables      map[string]any
	State          []StateDeclaration
	Schedulers     map[string]SchedulerDeclaration
	Graphs         map[string]GraphModule
	Steps          []Step
}

/*
Step is one runtime instruction.
*/
type Step struct {
	ID     string
	Op     string
	In     map[string]string
	Out    map[string]string
	Graph  string
	Config map[string]any
	Loop   *Loop
	Body   []Step
}

/*
Loop describes iteration over a collection or count.
*/
type Loop struct {
	Over   string
	As     string
	Repeat string
	Until  string
}

/*
StateDeclaration names a runtime state object and its initializer.
*/
type StateDeclaration struct {
	Name   string
	Type   string
	Shape  []any
	Init   string
	Seed   any
	Config map[string]any
}

/*
SchedulerDeclaration names a scheduler implementation and its config.
*/
type SchedulerDeclaration struct {
	Type   string
	Config map[string]any
}

/*
GraphModule exposes a compiled topology under a program-local graph name.
*/
type GraphModule struct {
	Topology    *Topology
	Outputs     map[string]string
	InputShapes map[string][]any
}
