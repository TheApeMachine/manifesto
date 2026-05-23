package runtime

import (
	"fmt"

	"github.com/theapemachine/manifesto/diffusion"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func uploadPackedLatents(
	memory tensor.Backend,
	layout diffusion.LatentLayout,
	seed int64,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	if memory == nil {
		return nil, fmt.Errorf("diffusion prepare latents: tensor backend is required")
	}

	values, err := diffusion.SamplePackedLatents(layout, seed)

	if err != nil {
		return nil, err
	}

	shape, err := tensor.NewShape([]int{1, layout.ImageSeqLen, layout.PackedChannels})

	if err != nil {
		return nil, err
	}

	return memory.Upload(shape, storageDType, Float32AsDTypeBytes(values, storageDType))
}
