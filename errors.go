package manifest

import "fmt"

/*
Error is a manifest compilation or execution failure with a stable path for
diagnostics.
*/
type Error struct {
	Path    string
	Op      string
	Message string
	Cause   error
}

func (manifestError *Error) Error() string {
	if manifestError.Path != "" {
		return fmt.Sprintf("manifest %s: %s: %v", manifestError.Path, manifestError.Message, manifestError.Unwrap())
	}

	return fmt.Sprintf("manifest: %s: %v", manifestError.Message, manifestError.Unwrap())
}

func (manifestError *Error) Unwrap() error {
	return manifestError.Cause
}

func newError(path, op, message string, cause error) *Error {
	return &Error{
		Path:    path,
		Op:      op,
		Message: message,
		Cause:   cause,
	}
}
