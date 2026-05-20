package hfconfig

import (
	"encoding/json"
	"fmt"
	"io"
)

/*
Config represents a standard Hugging Face config.json file.
It contains the architectural parameters needed to dynamically
generate a manifesto AST.
*/
type Config struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`

	// Common Transformer parameters
	VocabSize         int     `json:"vocab_size"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	MaxPositionEmbeds int     `json:"max_position_embeddings"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	LayerNormEps      float32 `json:"layer_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	AttentionBias     bool    `json:"attention_bias"`
	AttentionDropout  float32 `json:"attention_dropout"`
	HiddenAct         string  `json:"hidden_act"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`
	BosTokenID        any     `json:"bos_token_id"`
	EosTokenID        any     `json:"eos_token_id"`
}

/*
ParseConfig reads a Hugging Face config.json file and returns a Config struct.
*/
func ParseConfig(reader io.Reader) (*Config, error) {
	bytes, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("hfconfig parse: failed to read config: %w", err)
	}

	var config Config
	if err := json.Unmarshal(bytes, &config); err != nil {
		return nil, fmt.Errorf("hfconfig parse: failed to unmarshal json: %w", err)
	}

	// Set defaults for missing fields
	if config.NumKeyValueHeads == 0 {
		config.NumKeyValueHeads = config.NumAttentionHeads
	}
	if config.RopeTheta == 0 {
		config.RopeTheta = 10000.0
	}
	if config.RMSNormEps == 0 {
		config.RMSNormEps = 1e-5
	}
	if config.LayerNormEps == 0 {
		config.LayerNormEps = 1e-5
	}

	return &config, nil
}
