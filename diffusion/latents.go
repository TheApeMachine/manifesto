package diffusion

import (
	"fmt"
	"math/rand"
)

/*
PositionID identifies one packed latent token in row/column space.
*/
type PositionID struct {
	Row int32
	Col int32
}

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

/*
PrepareLatentIDs builds row/column identifiers for each packed latent token.
*/
func PrepareLatentIDs(batchSize int, packedHeight int, packedWidth int) ([]PositionID, error) {
	if batchSize <= 0 || packedHeight <= 0 || packedWidth <= 0 {
		return nil, fmt.Errorf("diffusion prepare latent ids: batch and packed dimensions must be positive")
	}

	sequenceLength := batchSize * packedHeight * packedWidth
	identifiers := make([]PositionID, 0, sequenceLength)

	for batchIndex := range batchSize {
		for rowIndex := range packedHeight {
			for columnIndex := range packedWidth {
				identifiers = append(identifiers, PositionID{
					Row: int32(rowIndex),
					Col: int32(columnIndex),
				})

				_ = batchIndex
			}
		}
	}

	return identifiers, nil
}
