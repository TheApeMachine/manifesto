package convert

import (
	"encoding/binary"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

func decodeFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = math.Float64frombits(binary.LittleEndian.Uint64(buf[index*8:]))
	}

	return out
}

func decodeFloat32ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[index*4:])))
	}

	return out
}

func decodeFloat16ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		value := dtype.Frombits(binary.LittleEndian.Uint16(buf[index*2:]))
		out[index] = float64(value.Float32())
	}

	return out
}

func decodeBFloat16ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		value := dtype.NewBfloat16FromBytes(buf[index*2:])
		out[index] = float64((&value).Float32())
	}

	return out
}

func decodeFloat8E4M3ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		value := dtype.F8E4M3(buf[index])
		out[index] = float64(value.Float32())
	}

	return out
}

func decodeFloat8E5M2ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		value := dtype.F8E5M2(buf[index])
		out[index] = float64(value.Float32())
	}

	return out
}

func decodeInt64ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		raw := binary.LittleEndian.Uint64(buf[index*8:])
		out[index] = float64(int64(raw))
	}

	return out
}

func decodeInt32ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		raw := binary.LittleEndian.Uint32(buf[index*4:])
		out[index] = float64(int32(raw))
	}

	return out
}

func decodeInt16ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		raw := binary.LittleEndian.Uint16(buf[index*2:])
		out[index] = float64(int16(raw))
	}

	return out
}

func decodeInt8ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(int8(buf[index]))
	}

	return out
}

func decodeUint64ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint64(buf[index*8:]))
	}

	return out
}

func decodeUint32ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint32(buf[index*4:]))
	}

	return out
}

func decodeUint16ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(binary.LittleEndian.Uint16(buf[index*2:]))
	}

	return out
}

func decodeUint8ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		out[index] = float64(buf[index])
	}

	return out
}

func decodeInt4ToFloat64(buf []byte, out []float64) []float64 {
	for index := range out {
		pair := dtype.Int4Pair(buf[index/2])

		if index%2 == 0 {
			out[index] = float64(pair.Lo())
			continue
		}

		out[index] = float64(pair.Hi())
	}

	return out
}
