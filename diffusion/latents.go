package diffusion

import (
	"fmt"
	"math/rand"
)

/*
PackLatents maps [batch, channels, height, width] to [batch, height*width, channels].
*/
func PackLatents(values []float32, batchSize int, channels int, height int, width int) ([]float32, error) {
	expected := batchSize * channels * height * width

	if len(values) != expected {
		return nil, fmt.Errorf(
			"diffusion pack latents: expected %d values, got %d",
			expected,
			len(values),
		)
	}

	packed := make([]float32, batchSize*height*width*channels)
	writeIndex := 0

	for batchIndex := range batchSize {
		for rowIndex := range height {
			for columnIndex := range width {
				for channelIndex := range channels {
					sourceIndex := (((batchIndex*channels)+channelIndex)*height+rowIndex)*width + columnIndex
					packed[writeIndex] = values[sourceIndex]
					writeIndex++
				}
			}
		}
	}

	return packed, nil
}

/*
SamplePackedLatents draws a Gaussian grid and packs it for the denoiser.
*/
func SamplePackedLatents(layout LatentLayout, seed int64) ([]float32, error) {
	rng := rand.New(rand.NewSource(seed))
	batchSize := 1
	channels := layout.PackedChannels
	height := layout.PackedHeight
	width := layout.PackedWidth
	values := make([]float32, batchSize*channels*height*width)

	for index := range values {
		values[index] = float32(rng.NormFloat64())
	}

	return PackLatents(values, batchSize, channels, height, width)
}
