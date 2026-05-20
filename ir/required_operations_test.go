package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRequiredOperationIDs(t *testing.T) {
	Convey("Given the required operation ID contract", t, func() {
		Convey("It should be non-empty and unique", func() {
			operationIDs := RequiredOperationIDs()

			So(operationIDs, ShouldNotBeEmpty)

			seen := make(map[OpType]bool, len(operationIDs))

			for _, operationID := range operationIDs {
				So(seen[operationID], ShouldBeFalse)

				seen[operationID] = true
			}
		})
	})
}
