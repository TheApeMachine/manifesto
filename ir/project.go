package ir

import "time"

/*
Project is the IR of a research project.
*/
type Project struct {
	Kind         Kind
	Name         string
	Description  string
	Created      *time.Time
	Updated      *time.Time
	Metadata     map[string]string
	Architecture *Architecture
}

/*
Node returns the topology node with the given checkpoint prefix name.
*/
func (project *Project) Node(name string) *Node {
	if project == nil || project.Architecture == nil || project.Architecture.Topology == nil {
		return nil
	}

	for _, node := range project.Architecture.Topology.Nodes {
		if node.Name == name {
			return node
		}
	}

	return nil
}
