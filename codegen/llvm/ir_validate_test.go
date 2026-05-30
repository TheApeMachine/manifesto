package llvm

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/optimizer"
)

func TestValidateFusionAST(testingObject *testing.T) {
	convey.Convey("Given an unsupported fusion node", testingObject, func() {
		fusion := &optimizer.FusionAST{
			InputPorts: []string{"x"},
			OutputPort: "bad",
			Root:       &optimizer.ASTNode{Type: optimizer.NodeInvalid},
		}

		_, err := BuildModule(fusion, "fused_bad")
		convey.So(err, convey.ShouldNotBeNil)
		convey.So(err.Error(), convey.ShouldContainSubstring, "unsupported fusion node")
	})
}
