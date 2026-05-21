package asset

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestWalk(t *testing.T) {
	Convey("Given embedded templates", t, func() {
		Convey("Walk", func() {
			Convey("It should return operation, block, and model schemas", func() {
				schemas, err := Walk("template/operation")

				So(err, ShouldBeNil)
				So(schemas, ShouldNotBeEmpty)

				blocks, err := Walk("template/block")

				So(err, ShouldBeNil)
				So(blocks, ShouldNotBeEmpty)

				models, err := Walk("template/model")

				So(err, ShouldBeNil)
				So(models, ShouldNotBeEmpty)
			})
		})
	})
}

func TestWalk_EnergyBlocks(t *testing.T) {
	Convey("Given energy block templates", t, func() {
		blocks, err := Walk("template/block")

		So(err, ShouldBeNil)

		operations, err := Walk("template/operation")

		So(err, ShouldBeNil)

		energyBlockIDs := []string{
			"block.energy.boltzmann_distribution",
			"block.energy.contrastive_phase",
			"block.energy.free_energy",
			"block.energy.langevin_step",
		}

		Convey("Walk", func() {
			Convey("It should expose executable primitive topologies", func() {
				for _, energyBlockID := range energyBlockIDs {
					schema, exists := blocks[energyBlockID]

					So(exists, ShouldBeTrue)
					So(schema.Kind, ShouldEqual, "Block")
					So(schema.Category, ShouldEqual, "energy")
					So(schema.System, ShouldNotBeNil)
					So(schema.System.Topology.Nodes, ShouldNotBeEmpty)

					for _, node := range schema.System.Topology.Nodes {
						So(node.ID, ShouldNotBeBlank)
						So(node.Op, ShouldNotBeBlank)
						So(node.Out, ShouldNotBeEmpty)

						_, operationExists := operations[node.Op]

						So(operationExists, ShouldBeTrue)
					}
				}
			})
		})
	})
}

func BenchmarkWalk_EnergyBlocks(benchmark *testing.B) {
	for benchmark.Loop() {
		blocks, err := Walk("template/block")

		if err != nil {
			benchmark.Fatal(err)
		}

		_ = blocks["block.energy.langevin_step"]
	}
}
