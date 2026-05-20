package tensor

/*
NUMA support surface. The platform-specific implementations live in
numa_linux.go (libnuma-backed via cgo) and numa_darwin.go (single-node
no-op). This file declares the portable contract.

Phase 3 ships the host-only NUMA path. Per spray-and-pray, the Linux
implementation here returns ErrNeedsPlatformSetup until the libnuma
cgo bindings land in a later session that can verify on multi-socket
hardware.
*/

/*
NUMAQuery reports the host's NUMA topology depth. Returns 1 on
single-node hosts (Darwin always; Linux without libnuma).
*/
func NUMAQuery() int {
	return numaNodes()
}

/*
NUMAPreferredNode returns the current goroutine's preferred NUMA node,
which the host allocator uses as the default placement target.
Returns 0 on single-node hosts.
*/
func NUMAPreferredNode() int {
	return numaPreferredNode()
}

/*
AllocateOn returns a 64-byte-aligned byte slice allocated on the given
NUMA node. On single-node hosts the node parameter is ignored.
*/
func AllocateOn(node int, bytesNeeded int) []byte {
	return numaAllocateOn(node, bytesNeeded)
}
