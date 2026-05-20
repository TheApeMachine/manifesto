package resolve

import (
	"context"
	"io"
)

/*
RepoType identifies a Hugging Face hub repository kind.
*/
type RepoType string

const (
	ModelRepo   RepoType = "model"
	DatasetRepo RepoType = "dataset"
	SpaceRepo   RepoType = "space"
)

/*
RepoLocation identifies a revision-pinned hub repository.
*/
type RepoLocation struct {
	RepoID   string
	RepoType RepoType
	Revision string
	Token    string
}

/*
DownloadRequest fetches one file from a hub repository.
*/
type DownloadRequest struct {
	Location RepoLocation
	Filename string
	CacheDir string
}

/*
File is an immutable local snapshot of one hub artifact.
*/
type File struct {
	Path   string
	Commit string
	Size   int64
}

/*
Hub resolves and downloads Hugging Face repository artifacts. Hosts implement
this interface; manifest never imports a concrete hub client.
*/
type Hub interface {
	Download(ctx context.Context, request DownloadRequest) (*File, error)
	ReadJSON(ctx context.Context, location RepoLocation, filename string, cacheDir string, target any) error
	Open(ctx context.Context, location RepoLocation, filename string, cacheDir string) (io.ReadCloser, *File, error)
	Glob(ctx context.Context, location RepoLocation, pattern string, cacheDir string) ([]string, error)
}
