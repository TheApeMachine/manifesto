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
	latents, _, err := uploadPackedLatentsWithIDs(memory, layout, seed, storageDType)

	return latents, err
}

func uploadPackedLatentsWithIDs(
	memory tensor.Backend,
	layout diffusion.LatentLayout,
	seed int64,
	storageDType dtype.DType,
) (tensor.Tensor, []diffusion.PositionID, error) {
	if memory == nil {
		return nil, nil, fmt.Errorf("diffusion prepare latents: tensor backend is required")
	}

	values, err := diffusion.SamplePackedLatents(layout, seed)

	if err != nil {
		return nil, nil, err
	}

	shape, err := tensor.NewShape([]int{1, layout.ImageSeqLen, layout.PackedChannels})

	if err != nil {
		return nil, nil, err
	}

	latents, err := memory.Upload(shape, storageDType, Float32AsDTypeBytes(values, storageDType))

	if err != nil {
		return nil, nil, err
	}

	latentIDs, err := diffusion.PrepareLatentIDs(1, layout.PackedHeight, layout.PackedWidth)

	if err != nil {
		_ = latents.Close()
		return nil, nil, err
	}

	return latents, latentIDs, nil
}
