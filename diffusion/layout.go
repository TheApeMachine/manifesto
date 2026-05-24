package diffusion

import "fmt"

/*
LatentLayout holds resolution-derived token grid sizes for packed latent denoisers.
All inputs are manifest-supplied (generation.width/height/latent_downsample/latent_channels).
*/
type LatentLayout struct {
	LatentDownsample int
	SnappedHeight    int
	SnappedWidth     int
	PackedHeight     int
	PackedWidth      int
	ImageSeqLen      int
	LatentSide       int
	VAESpatial       int
	MidAttnTokens    int
	PackedChannels   int
}

/*
ComputeLatentLayout derives denoiser/VAE spatial variables from generation manifest fields.
*/
func ComputeLatentLayout(
	width int,
	height int,
	latentDownsample int,
	packedChannels int,
) (LatentLayout, error) {
	if width <= 0 || height <= 0 {
		return LatentLayout{}, fmt.Errorf("diffusion layout: width and height must be positive")
	}

	if latentDownsample <= 0 {
		return LatentLayout{}, fmt.Errorf("diffusion layout: latent_downsample must be positive")
	}

	if packedChannels <= 0 {
		return LatentLayout{}, fmt.Errorf("diffusion layout: latent_channels must be positive")
	}

	snappedHeight := 2 * (height / latentDownsample)
	snappedWidth := 2 * (width / latentDownsample)

	if snappedHeight <= 0 || snappedWidth <= 0 {
		return LatentLayout{}, fmt.Errorf(
			"diffusion layout: %dx%d is too small for latent_downsample %d",
			width,
			height,
			latentDownsample,
		)
	}

	packedHeight := snappedHeight / 2
	packedWidth := snappedWidth / 2

	return LatentLayout{
		LatentDownsample: latentDownsample,
		SnappedHeight:    snappedHeight,
		SnappedWidth:     snappedWidth,
		PackedHeight:     packedHeight,
		PackedWidth:      packedWidth,
		ImageSeqLen:      packedHeight * packedWidth,
		LatentSide:       packedHeight,
		VAESpatial:       snappedHeight,
		MidAttnTokens:    snappedHeight * snappedWidth,
		PackedChannels:   packedChannels,
	}, nil
}
