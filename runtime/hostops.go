package runtime

import "context"

/*
HostOps supplies platform IO and tokenizer behavior to the program executor.
Hosts implement this interface; manifesto never imports hub or terminal packages.
*/
type HostOps interface {
	ReadLine(ctx context.Context) (string, error)
	EmitToken(ctx context.Context, request EmitTokenRequest) error
	WriteImage(ctx context.Context, request WriteImageRequest) error
	Encode(ctx context.Context, request EncodeRequest) ([]int, error)
}

/*
EncodeRequest identifies a tokenizer for one encode step.
*/
type EncodeRequest struct {
	Tokenizer         string
	TokenizerFile     string
	Text              string
	ApplyChatTemplate bool
	ChatContinuation  bool
	MaxLength         int
	PadTokenID        int
}

/*
EmitTokenRequest identifies a tokenizer for one emit step.
*/
type EmitTokenRequest struct {
	Tokenizer     string
	TokenizerFile string
	TokenID       int
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
	Layout   string
	Range    string
}
