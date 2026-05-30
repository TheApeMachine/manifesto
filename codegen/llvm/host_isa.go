package llvm

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

/*
ISALevel names the host SIMD tier selected for LLVM JIT codegen.
*/
type ISALevel int

const (
	ISALevelSSE2 ISALevel = iota
	ISALevelAVX2
	ISALevelAVX512
	ISALevelNEON
)

/*
HostISALevel returns the highest SIMD tier available on the running host.
*/
func HostISALevel() ISALevel {
	switch runtime.GOARCH {
	case "amd64":
		if cpu.X86.HasAVX512F {
			return ISALevelAVX512
		}

		if cpu.X86.HasAVX2 {
			return ISALevelAVX2
		}

		return ISALevelSSE2
	case "arm64":
		return ISALevelNEON
	default:
		return ISALevelSSE2
	}
}

/*
HostVectorWidth returns the number of float32 lanes for explicit vector IR
on the host ISA tier.
*/
func HostVectorWidth() int {
	switch HostISALevel() {
	case ISALevelAVX512:
		return 16
	case ISALevelAVX2:
		return 8
	case ISALevelNEON, ISALevelSSE2:
		return 4
	default:
		return 4
	}
}

/*
CPUFeaturesForLevel returns LLVM target feature flags for one ISA tier.
*/
func CPUFeaturesForLevel(level ISALevel) string {
	switch level {
	case ISALevelAVX512:
		return "+avx512f,+avx2,+sse2"
	case ISALevelAVX2:
		return "+avx2,+sse2"
	case ISALevelNEON:
		return "+neon"
	case ISALevelSSE2:
		return "+sse2"
	default:
		return "+sse2"
	}
}

/*
SupportedISALevelsOnHost returns ISA tiers this host can execute, highest first.
*/
func SupportedISALevelsOnHost() []ISALevel {
	switch runtime.GOARCH {
	case "amd64":
		levels := make([]ISALevel, 0, 3)

		if cpu.X86.HasAVX512F {
			levels = append(levels, ISALevelAVX512)
		}

		if cpu.X86.HasAVX2 {
			levels = append(levels, ISALevelAVX2)
		}

		levels = append(levels, ISALevelSSE2)

		return levels
	case "arm64":
		return []ISALevel{ISALevelNEON}
	default:
		return []ISALevel{ISALevelSSE2}
	}
}

func (level ISALevel) String() string {
	switch level {
	case ISALevelSSE2:
		return "sse2"
	case ISALevelAVX2:
		return "avx2"
	case ISALevelAVX512:
		return "avx512"
	case ISALevelNEON:
		return "neon"
	default:
		return "unknown"
	}
}
