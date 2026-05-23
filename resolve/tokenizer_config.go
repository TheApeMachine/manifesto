package resolve

import (
	"context"
	"fmt"
)

/*
TokenizerConfig loads tokenizer/tokenizer_config.json for one model repository.
*/
func (resolver *Resolver) TokenizerConfig(
	ctx context.Context,
	location RepoLocation,
	cacheDir string,
) (map[string]any, error) {
	config := make(map[string]any)

	err := resolver.hub.ReadJSON(ctx, location, "tokenizer/tokenizer_config.json", cacheDir, &config)

	if err != nil {
		return nil, fmt.Errorf("resolve tokenizer config: %w", err)
	}

	return config, nil
}

/*
TokenizerVariables derives runtime tokenizer fields from a Hub tokenizer_config.json.
*/
func TokenizerVariables(tokenizerConfig map[string]any) map[string]any {
	variables := make(map[string]any)

	if modelMaxLength := intFromHubConfig(tokenizerConfig["model_max_length"]); modelMaxLength > 0 {
		variables["model_max_length"] = modelMaxLength
	}

	if padTokenID, ok := tokenIDFromTokenizerConfig(tokenizerConfig, "pad_token"); ok {
		variables["pad_token_id"] = padTokenID
	}

	if eosTokenID, ok := tokenIDFromTokenizerConfig(tokenizerConfig, "eos_token"); ok {
		variables["eos_token_id"] = eosTokenID
	}

	if bosTokenID, ok := tokenIDFromTokenizerConfig(tokenizerConfig, "bos_token"); ok {
		variables["bos_token_id"] = bosTokenID
	}

	return variables
}

func tokenIDFromTokenizerConfig(tokenizerConfig map[string]any, field string) (int, bool) {
	tokenText := tokenStringFromConfig(tokenizerConfig[field])

	if tokenText == "" {
		return 0, false
	}

	addedTokens, ok := tokenizerConfig["added_tokens_decoder"].(map[string]any)

	if !ok {
		return 0, false
	}

	for rawID, rawEntry := range addedTokens {
		entry, ok := rawEntry.(map[string]any)

		if !ok {
			continue
		}

		content, ok := entry["content"].(string)

		if !ok || content != tokenText {
			continue
		}

		tokenID := intFromHubConfig(rawID)

		if tokenID >= 0 {
			return tokenID, true
		}
	}

	return 0, false
}

func tokenStringFromConfig(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case map[string]any:
		content, ok := typed["content"].(string)

		if ok {
			return content
		}
	}

	return ""
}

func intFromHubConfig(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		parsed := 0

		for _, character := range typed {
			if character < '0' || character > '9' {
				return -1
			}

			parsed = parsed*10 + int(character-'0')
		}

		return parsed
	default:
		return -1
	}
}
