//go:build linux

package tensor

/*
Linux NUMA bindings. The production implementation cgo-links libnuma
and uses numa_alloc_onnode / numa_run_on_node. Per spray-and-pray,
this file currently provides a single-node fallback; the cgo bindings
are tagged for a later session that can be verified on a real
multi-socket box.

The fallback returns plain mmap'd buffers and treats every request as
node 0, so callers compile and run on single-node Linux without
libnuma installed.
*/

func numaNodes() int {
	return 1
}

func numaPreferredNode() int {
	return 0
}

func numaAllocateOn(node int, bytesNeeded int) []byte {
	buffer, err := Allocate(bytesNeeded)

	if err != nil {
		return nil
	}

	return buffer
}
