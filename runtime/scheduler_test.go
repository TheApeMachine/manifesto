package runtime

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestFlowMatchEulerDiscreteTimesteps(testingObject *testing.T) {
	convey.Convey("Given a FLUX2 dynamic shifting scheduler", testingObject, func() {
		scheduler, err := NewFlowMatchEulerDiscrete(SchedulerConfig{
			Steps:             50,
			NumTrainTimesteps: 1000,
			Shift:             3,
			UseDynamicShift:   true,
			TimeShiftType:     "exponential",
			ImageSeqLen:       4096,
		})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should match the HF sigma-derived timestep schedule", func() {
			timesteps := scheduler.Timesteps()
			expected := expectedFlux2Timestep(4096, 50, 0)
			last := expectedFlux2Timestep(4096, 50, 49)

			convey.So(timesteps, convey.ShouldHaveLength, 50)
			convey.So(math.Abs(float64(timesteps[0]-expected)), convey.ShouldBeLessThan, 1e-4)
			convey.So(math.Abs(float64(timesteps[49]-last)), convey.ShouldBeLessThan, 1e-4)
		})
	})
}

func TestFlowMatchEulerDiscreteDelta(testingObject *testing.T) {
	convey.Convey("Given a flow-match Euler scheduler", testingObject, func() {
		scheduler, err := NewFlowMatchEulerDiscrete(SchedulerConfig{
			Steps:             4,
			NumTrainTimesteps: 1000,
			Shift:             1,
			UseDynamicShift:   false,
			TimeShiftType:     "exponential",
			ImageSeqLen:       4096,
		})

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should expose the sigma interval as data", func() {
			convey.So(scheduler.Delta(1000), convey.ShouldEqual, float32(-0.25))
		})
	})
}

func expectedFlux2Timestep(imageSeqLen int, steps int, index int) float32 {
	sigma := 1.0 - (1.0-float64(1)/float64(steps))*float64(index)/float64(steps-1)
	mu := expectedFlux2Mu(imageSeqLen, steps)
	expMu := math.Exp(mu)

	return float32(1000 * expMu / (expMu + (1/sigma - 1)))
}

func expectedFlux2Mu(imageSeqLen int, steps int) float64 {
	a1, b1 := 8.73809524e-05, 1.89833333
	a2, b2 := 0.00016927, 0.45666666
	imageSeqLenFloat := float64(imageSeqLen)

	if imageSeqLen > 4300 {
		return a2*imageSeqLenFloat + b2
	}

	m200 := a2*imageSeqLenFloat + b2
	m10 := a1*imageSeqLenFloat + b1
	slope := (m200 - m10) / 190
	intercept := m200 - 200*slope

	return slope*float64(steps) + intercept
}
