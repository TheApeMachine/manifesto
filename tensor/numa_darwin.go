//go:build darwin

package tensor

/*
Darwin runs as a single NUMA node on Apple silicon and on current Intel
Macs. All NUMA calls are no-ops; AllocateOn ignores the node argument
and routes through the normal tiered allocator.
*/
func numaNodes() int {
	return 1
}

func numaPreferredNode() int {
	return 0
}

func numaAllocateOn(_ int, bytesNeeded int) []byte {
	buffer, err := Allocate(bytesNeeded)

	if err != nil {
		return nil
	}

	return buffer
}
