package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

/*
TensorMeta describes one SafeTensors entry.
*/
type TensorMeta struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

/*
IndexFile reads a SafeTensors header from disk.
*/
func IndexFile(path string) (map[string]TensorMeta, int64, error) {
	file, err := os.Open(path)

	if err != nil {
		return nil, 0, err
	}

	defer file.Close()

	return IndexReader(file)
}

/*
IndexReader reads a SafeTensors header from any reader.
*/
func IndexReader(reader io.Reader) (map[string]TensorMeta, int64, error) {
	var headerLength uint64

	if err := binary.Read(reader, binary.LittleEndian, &headerLength); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	headerBytes := make([]byte, headerLength)

	if _, err := io.ReadFull(reader, headerBytes); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	raw := make(map[string]json.RawMessage)

	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	index := make(map[string]TensorMeta, len(raw))

	for name, rawMeta := range raw {
		if name == "__metadata__" {
			continue
		}

		meta := TensorMeta{}

		if err := json.Unmarshal(rawMeta, &meta); err != nil {
			return nil, 0, fmt.Errorf("safetensors tensor %q: %w", name, err)
		}

		index[name] = meta
	}

	return index, int64(headerLength) + 8, nil
}

/*
ReadTensor returns the raw tensor bytes for one SafeTensors entry.
*/
func ReadTensor(path string, tensorName string) ([]byte, TensorMeta, error) {
	index, dataBase, err := IndexFile(path)

	if err != nil {
		return nil, TensorMeta{}, err
	}

	meta, ok := index[tensorName]

	if !ok {
		return nil, TensorMeta{}, fmt.Errorf("safetensors: missing tensor %q", tensorName)
	}

	file, err := os.Open(path)

	if err != nil {
		return nil, TensorMeta{}, err
	}

	defer file.Close()

	start := dataBase + meta.DataOffsets[0]
	length := meta.DataOffsets[1] - meta.DataOffsets[0]
	buffer := make([]byte, length)

	if _, err := file.ReadAt(buffer, start); err != nil {
		return nil, TensorMeta{}, fmt.Errorf("safetensors read %q: %w", tensorName, err)
	}

	return buffer, meta, nil
}
