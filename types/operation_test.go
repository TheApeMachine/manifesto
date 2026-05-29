package types

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestOpBindMethodResolvesSchema(t *testing.T) {
	convey.Convey("Given the embedded operation registry", t, func() {
		registry, err := NewOperationRegistry()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should resolve bind.method for manifest ops", func() {
			method, err := Op("activation.gelu").BindMethod(registry)
			convey.So(err, convey.ShouldBeNil)
			convey.So(method, convey.ShouldEqual, "Gelu")

			method, err = Op("projection.linear").BindMethod(registry)
			convey.So(err, convey.ShouldBeNil)
			convey.So(method, convey.ShouldEqual, "Matmul")

			method, err = Op("math.rmsnorm").BindMethod(registry)
			convey.So(err, convey.ShouldBeNil)
			convey.So(method, convey.ShouldEqual, "RMSNorm")
		})
	})
}

func TestNewOperationRegistryLoadsSchemas(t *testing.T) {
	convey.Convey("Given NewOperationRegistry", t, func() {
		registry, err := NewOperationRegistry()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should register activation.gelu", func() {
			schema, ok := registry.Lookup(Op("activation.gelu"))
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(schema.Op, convey.ShouldEqual, "activation.gelu")
			convey.So(schema.Bind.Method, convey.ShouldEqual, "Gelu")
		})
	})
}

func BenchmarkOpBindMethod(b *testing.B) {
	registry, err := NewOperationRegistry()

	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		if _, err := Op("activation.gelu").BindMethod(registry); err != nil {
			b.Fatal(err)
		}
	}
}
