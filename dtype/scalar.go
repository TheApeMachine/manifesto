package dtype

import (
	"encoding/json"
	"fmt"
	"math"
)

/*
Int64Value coerces JSON-decoded configuration scalars into int64. Manifest
recipe bindings use this for layer counts, hidden sizes, and other whole
number dimensions.
*/
func Int64Value(value any) (int64, error) {
	switch typed := value.(type) {
	case int:
		return int64(typed), nil
	case int32:
		return int64(typed), nil
	case int64:
		return typed, nil
	case float32:
		return floatToInt64(float64(typed))
	case float64:
		return floatToInt64(typed)
	case json.Number:
		asInt, err := typed.Int64()

		if err == nil {
			return asInt, nil
		}

		asFloat, floatErr := typed.Float64()

		if floatErr != nil {
			return 0, fmt.Errorf("dtype: parse json number %q: %w", typed, floatErr)
		}

		return floatToInt64(asFloat)
	default:
		return 0, fmt.Errorf("dtype: unsupported int64 scalar %T", value)
	}
}

/*
Float64Value coerces JSON-decoded configuration scalars into float64 for
hyperparameters such as rope theta or epsilon.
*/
func Float64Value(value any) (float64, error) {
	switch typed := value.(type) {
	case int:
		return float64(typed), nil
	case int32:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	case float32:
		return float64(typed), nil
	case float64:
		return typed, nil
	case json.Number:
		return typed.Float64()
	default:
		return 0, fmt.Errorf("dtype: unsupported float64 scalar %T", value)
	}
}

func floatToInt64(value float64) (int64, error) {
	truncated := math.Trunc(value)

	if truncated != value {
		return 0, fmt.Errorf("dtype: non-integer float scalar %v", value)
	}

	if truncated > float64(math.MaxInt64) || truncated < float64(math.MinInt64) {
		return 0, fmt.Errorf("dtype: scalar %v overflows int64", value)
	}

	return int64(truncated), nil
}
