package runtime

import "context"

/*
HostOps supplies platform IO and tokenizer behavior to the program executor.
Hosts implement this interface; manifesto never imports hub or terminal packages.
*/
type HostOps interface {
	ReadLine(ctx context.Context) (string, error)
	EmitToken(ctx context.Context, tokenID int) error
	WriteImage(ctx context.Context, request WriteImageRequest) error
	Encode(ctx context.Context, request EncodeRequest) ([]int, error)
}

/*
EncodeRequest identifies a tokenizer for one encode step.
*/
type EncodeRequest struct {
	Tokenizer string
	Text      string
}

/*
WriteImageRequest describes an image artifact write.
*/
type WriteImageRequest struct {
	Path     string
	Tensor   any
	Width    int
	Height   int
	Channels int
}
