package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
)

const (
	khronosBaseURL  = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
	khronosIndexURL = khronosBaseURL + "/model-index.json"
)

type khronosIndexStatus int

const (
	khronosIdle khronosIndexStatus = iota
	khronosLoading
	khronosLoaded
	khronosFailed
)

type khronosRawEntry struct {
	Label    string            `json:"label"`
	Name     string            `json:"name"`
	Variants map[string]string `json:"variants"`
}

// KhronosEntry is a single browsable model from the glTF-Sample-Assets index.
type KhronosEntry struct {
	Label              string
	Name               string
	GlbURL             string
	GltfFolderFilename string
}

// PendingModel is a downloaded model awaiting upload on the main thread.
type PendingModel struct {
	DisplayName string
	GltfBytes   []byte
	Resources   map[string][]byte
}

// KhronosBrowser fetches the Khronos glTF-Sample-Assets index and individual
type KhronosBrowser struct {
	mu       sync.Mutex
	status   khronosIndexStatus
	entries  []KhronosEntry
	indexErr string
	pending  *PendingModel
	loading  string
}

func NewKhronosBrowser() *KhronosBrowser {
	return &KhronosBrowser{}
}

// EnsureLoaded kicks off the index fetch the first time it is called.
func (b *KhronosBrowser) EnsureLoaded() {
	b.mu.Lock()
	if b.status != khronosIdle {
		b.mu.Unlock()
		return
	}
	b.status = khronosLoading
	b.mu.Unlock()
	go b.fetchIndex()
}

// Entries returns the loaded asset list, or nil while loading or on failure.
func (b *KhronosBrowser) Entries() []KhronosEntry {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.status != khronosLoaded {
		return nil
	}
	return b.entries
}

// IndexError returns the index fetch failure message, or "" if none.
func (b *KhronosBrowser) IndexError() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.status == khronosFailed {
		return b.indexErr
	}
	return ""
}

// LoadingStatus returns the display name of the model currently downloading.
func (b *KhronosBrowser) LoadingStatus() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.loading
}

// TakePending removes and returns a finished download, or nil if none is ready.
func (b *KhronosBrowser) TakePending() *PendingModel {
	b.mu.Lock()
	defer b.mu.Unlock()
	pending := b.pending
	b.pending = nil
	return pending
}

// FetchEntry begins downloading the given entry.
func (b *KhronosBrowser) FetchEntry(entry KhronosEntry) {
	if entry.GlbURL != "" {
		b.startGlbFetch(entry.Label, entry.GlbURL)
		return
	}
	if entry.GltfFolderFilename != "" {
		b.startGltfFolderFetch(entry.Name, entry.Label, entry.GltfFolderFilename)
	}
}

func (b *KhronosBrowser) beginLoading(displayName string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.loading != "" {
		return false
	}
	b.loading = displayName
	return true
}

func (b *KhronosBrowser) clearLoading() {
	b.mu.Lock()
	b.loading = ""
	b.mu.Unlock()
}

func (b *KhronosBrowser) publishPending(pending *PendingModel) {
	b.mu.Lock()
	b.pending = pending
	b.loading = ""
	b.mu.Unlock()
}

func (b *KhronosBrowser) fetchIndex() {
	data, ok := httpGetBytes(khronosIndexURL)
	if !ok {
		b.failIndex("fetch index")
		return
	}
	var raw []khronosRawEntry
	if err := json.Unmarshal(data, &raw); err != nil {
		b.failIndex("parse: " + err.Error())
		return
	}
	entries := khronosEntriesFromRaw(raw)
	b.mu.Lock()
	b.entries = entries
	b.status = khronosLoaded
	b.mu.Unlock()
}

func (b *KhronosBrowser) failIndex(message string) {
	b.mu.Lock()
	b.status = khronosFailed
	b.indexErr = message
	b.mu.Unlock()
}

func (b *KhronosBrowser) startGlbFetch(displayName, glbURL string) {
	if !b.beginLoading(displayName) {
		return
	}
	go func() {
		data, ok := httpGetBytes(glbURL)
		if !ok {
			b.clearLoading()
			return
		}
		b.publishPending(&PendingModel{DisplayName: displayName, GltfBytes: data})
	}()
}

func (b *KhronosBrowser) startGltfFolderFetch(assetName, displayName, gltfFilename string) {
	if !b.beginLoading(displayName) {
		return
	}
	folderURL := khronosBaseURL + "/" + assetName + "/glTF"
	gltfURL := folderURL + "/" + gltfFilename
	go func() {
		gltfBytes, ok := httpGetBytes(gltfURL)
		if !ok {
			b.clearLoading()
			return
		}
		uris := externalURIsFromGltf(gltfBytes)
		resources := make(map[string][]byte, len(uris))
		for _, uri := range uris {
			data, got := httpGetBytes(folderURL + "/" + uri)
			if !got {
				b.clearLoading()
				return
			}
			key := uri
			if decoded, err := url.PathUnescape(uri); err == nil {
				key = decoded
			}
			resources[key] = data
		}
		b.publishPending(&PendingModel{
			DisplayName: displayName,
			GltfBytes:   gltfBytes,
			Resources:   resources,
		})
	}()
}

func khronosEntriesFromRaw(raw []khronosRawEntry) []KhronosEntry {
	entries := make([]KhronosEntry, 0, len(raw))
	for _, r := range raw {
		entry := KhronosEntry{Label: r.Label, Name: r.Name}
		if filename, ok := r.Variants["glTF-Binary"]; ok {
			entry.GlbURL = khronosBaseURL + "/" + r.Name + "/glTF-Binary/" + filename
		}
		if filename, ok := r.Variants["glTF"]; ok {
			entry.GltfFolderFilename = filename
		}
		if entry.GlbURL == "" && entry.GltfFolderFilename == "" {
			continue
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool {
		return strings.ToLower(entries[i].Label) < strings.ToLower(entries[j].Label)
	})
	return entries
}

func externalURIsFromGltf(gltfBytes []byte) []string {
	var doc struct {
		Buffers []struct {
			URI string `json:"uri"`
		} `json:"buffers"`
		Images []struct {
			URI string `json:"uri"`
		} `json:"images"`
	}
	if err := json.Unmarshal(gltfBytes, &doc); err != nil {
		return nil
	}
	var uris []string
	for _, buffer := range doc.Buffers {
		if buffer.URI != "" && !strings.HasPrefix(buffer.URI, "data:") {
			uris = append(uris, buffer.URI)
		}
	}
	for _, image := range doc.Images {
		if image.URI != "" && !strings.HasPrefix(image.URI, "data:") {
			uris = append(uris, image.URI)
		}
	}
	return uris
}

func httpGetBytes(target string) ([]byte, bool) {
	resp, err := http.Get(target)
	if err != nil {
		log.Printf("http get %s: %v", target, err)
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("http get %s: HTTP %d", target, resp.StatusCode)
		return nil, false
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("http read %s: %v", target, err)
		return nil, false
	}
	return data, true
}
