package llvm

import (
	"fmt"
	"runtime"
)

/*
HostTargetTriple returns the LLVM target triple for the running host.
*/
func HostTargetTriple() string {
	switch runtime.GOARCH {
	case "amd64":
		switch runtime.GOOS {
		case "darwin":
			return "x86_64-apple-darwin"
		case "linux":
			return "x86_64-unknown-linux-gnu"
		case "windows":
			return "x86_64-pc-windows-msvc"
		default:
			return "x86_64-unknown-linux-gnu"
		}
	case "arm64":
		switch runtime.GOOS {
		case "darwin":
			return "arm64-apple-darwin"
		case "linux":
			return "aarch64-unknown-linux-gnu"
		default:
			return "aarch64-unknown-linux-gnu"
		}
	default:
		return ""
	}
}

/*
HostCPUFeatures returns LLVM target feature flags for the host's highest SIMD
tier. See CPUFeaturesForLevel and HostISALevel.
*/
func HostCPUFeatures() string {
	return CPUFeaturesForLevel(HostISALevel())
}

/*
ValidateHostJITSupport returns an error when the host cannot run the LLVM
JIT path (unknown triple, etc.).
*/
func ValidateHostJITSupport() error {
	if HostTargetTriple() == "" {
		return fmt.Errorf("codegen llvm: unsupported host GOARCH %q", runtime.GOARCH)
	}

	return nil
}
