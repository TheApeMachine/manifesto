package types

import "github.com/theapemachine/manifesto/dtype"

/*
Kind distinguishes metadata entries from tensor declarations in an archive index.
*/
type Kind uint8

const (
	KindMetadata Kind = iota + 1
	KindTensor
)

/*
Role classifies a tensor by what it does inside a model. The classifier
sets one Role per KindTensor token at parse time from (name, shape,
dtype); the downstream dispatcher uses the Role to decide which
device.Backend method (and which argument slot) the tensor's payload
plugs into.

Roles cover only the weighted device.Backend methods declared in
puter/device/interface.go. Weightless ops (sinusoidal timestep embed,
softmax, masking, RoPE rotation, residual add, …) have no token and
are invoked from the architecture recipe, not via Role.

Attention-specific roles (AttentionQKV, AttentionQKVMLP, AttentionOut,
AttentionQNorm, AttentionKNorm) are still Matmul or RMSNorm under the
hood — the role only tells the dispatcher which slot of the attention
call the pointer fills.

RoleUnknown is the zero value and means the classifier did not
recognize the tensor. Unknown tensors are not a parse error — the
dispatcher rejects them at wiring time with a clear diagnostic.
*/
type Role uint16

const (
	RoleUnknown Role = iota

	// Generic linear projection weight, rank-2 [out, in], used as
	// device.Matmul.Matmul's right operand.
	RoleLinearWeight
	// Rank-1 bias paired with a linear projection.
	RoleBias

	// Token-id lookup table, rank-2 [vocab, hidden], used as
	// device.Embedding.Lookup's table argument.
	RoleEmbeddingTable

	// Rank-1 affine scale for LayerNorm/RMSNorm/GroupNorm/InstanceNorm.
	RoleNormScale
	// Rank-1 affine bias for LayerNorm/GroupNorm/InstanceNorm.
	RoleNormBias
	// Running statistics for BatchNormEval.
	RoleNormMean
	RoleNormVariance

	// Convolution kernel — rank-3 (Conv1D), rank-4 (Conv2D), rank-5 (Conv3D).
	RoleConvKernel

	// Attention-family aliases. Still routed through Matmul or RMSNorm,
	// but the dispatcher needs to know which attention slot they fill.
	RoleAttentionQKV       // fused Q/K/V projection
	RoleAttentionQKVMLP    // fused QKV + MLP gate (FLUX-2 single-stream)
	RoleAttentionOut       // attention output projection
	RoleAttentionQNorm     // QK-Norm scale for Q
	RoleAttentionKNorm     // QK-Norm scale for K
	RoleModulation         // AdaLN-style modulation projection
	RoleProjectionOut      // final output / lm_head / proj_out
)

/*
String returns a stable human-readable label for the Role, used in
diagnostic messages and tests.
*/
func (role Role) String() string {
	switch role {
	case RoleUnknown:
		return "Unknown"
	case RoleLinearWeight:
		return "LinearWeight"
	case RoleBias:
		return "Bias"
	case RoleEmbeddingTable:
		return "EmbeddingTable"
	case RoleNormScale:
		return "NormScale"
	case RoleNormBias:
		return "NormBias"
	case RoleNormMean:
		return "NormMean"
	case RoleNormVariance:
		return "NormVariance"
	case RoleConvKernel:
		return "ConvKernel"
	case RoleAttentionQKV:
		return "AttentionQKV"
	case RoleAttentionQKVMLP:
		return "AttentionQKVMLP"
	case RoleAttentionOut:
		return "AttentionOut"
	case RoleAttentionQNorm:
		return "AttentionQNorm"
	case RoleAttentionKNorm:
		return "AttentionKNorm"
	case RoleModulation:
		return "Modulation"
	case RoleProjectionOut:
		return "ProjectionOut"
	default:
		return "Unknown"
	}
}

/*
Token is one entry from a serialized weight archive before payload bytes are read.

For KindTensor, Name is the full checkpoint key. Topology manifests bind graph
nodes to these names through weight entries, optionally with slice ranges when
one fused tensor feeds multiple nodes.

For KindMetadata, Name is the metadata key and Value is the metadata string.
Shape, Precision, Span, and Role are unused.
*/
type Token struct {
	Kind      Kind
	Name      string
	Value     string
	Shape     []int64
	Precision dtype.DType
	Span      Span
	Role      Role
}

/*
Span locates tensor payload bytes relative to the start of the archive data buffer.
*/
type Span struct {
	Offset int64
	Length int64
}
