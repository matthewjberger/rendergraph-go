package main

import (
	"encoding/json"
	"log"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const (
	polyhavenAssetsURL           = "https://api.polyhaven.com/assets?type=models"
	polyhavenFilesURL            = "https://api.polyhaven.com/files/"
	polyhavenPreferredResolution = 1
)

type polyhavenIndexStatus int

const (
	polyhavenIdle polyhavenIndexStatus = iota
	polyhavenLoading
	polyhavenLoaded
	polyhavenFailed
)

type polyhavenRawAsset struct {
	Name string `json:"name"`
}

// PolyhavenEntry is a single model from the Poly Haven asset index.
type PolyhavenEntry struct {
	Slug string
	Name string
}

type polyhavenIncludeFile struct {
	URL string `json:"url"`
}

type polyhavenGltfFile struct {
	URL     string                          `json:"url"`
	Include map[string]polyhavenIncludeFile `json:"include"`
}

type polyhavenGltfResolution struct {
	Gltf polyhavenGltfFile `json:"gltf"`
}

type polyhavenModelFiles struct {
	Gltf map[string]polyhavenGltfResolution `json:"gltf"`
}

// PolyhavenBrowser fetches the Poly Haven model index and downloads individual
// glTF models together with their sidecar resources off the main thread.
type PolyhavenBrowser struct {
	mu         sync.Mutex
	status     polyhavenIndexStatus
	entries    []PolyhavenEntry
	loading    string
	pending    *PendingModel
	wantRandom bool
}

func NewPolyhavenBrowser() *PolyhavenBrowser {
	return &PolyhavenBrowser{}
}

// TakePending removes and returns a finished download, or nil if none is ready.
func (b *PolyhavenBrowser) TakePending() *PendingModel {
	b.mu.Lock()
	defer b.mu.Unlock()
	pending := b.pending
	b.pending = nil
	return pending
}

// FetchRandom downloads a random model, kicking off the index fetch first if it
// has not loaded yet and deferring the pick until the index arrives.
func (b *PolyhavenBrowser) FetchRandom() {
	b.mu.Lock()
	switch b.status {
	case polyhavenIdle:
		b.status = polyhavenLoading
		b.wantRandom = true
		b.mu.Unlock()
		go b.fetchIndex()
		return
	case polyhavenLoading:
		b.wantRandom = true
		b.mu.Unlock()
		return
	case polyhavenFailed:
		b.mu.Unlock()
		return
	}
	entries := b.entries
	b.mu.Unlock()
	b.fetchRandomFrom(entries)
}

func (b *PolyhavenBrowser) fetchRandomFrom(entries []PolyhavenEntry) {
	if len(entries) == 0 {
		return
	}
	b.FetchEntry(entries[rand.IntN(len(entries))])
}

// FetchEntry begins downloading the given model entry.
func (b *PolyhavenBrowser) FetchEntry(entry PolyhavenEntry) {
	if !b.beginLoading(entry.Name) {
		return
	}
	go b.fetchModel(entry)
}

func (b *PolyhavenBrowser) beginLoading(displayName string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.loading != "" {
		return false
	}
	b.loading = displayName
	return true
}

func (b *PolyhavenBrowser) clearLoading() {
	b.mu.Lock()
	b.loading = ""
	b.mu.Unlock()
}

func (b *PolyhavenBrowser) publishPending(pending *PendingModel) {
	b.mu.Lock()
	b.pending = pending
	b.loading = ""
	b.mu.Unlock()
}

func (b *PolyhavenBrowser) fetchIndex() {
	data, ok := httpGetBytes(polyhavenAssetsURL)
	if !ok {
		b.failIndex("fetch index")
		return
	}
	var raw map[string]polyhavenRawAsset
	if err := json.Unmarshal(data, &raw); err != nil {
		b.failIndex("parse: " + err.Error())
		return
	}
	entries := polyhavenEntriesFromRaw(raw)
	b.mu.Lock()
	b.entries = entries
	b.status = polyhavenLoaded
	wantRandom := b.wantRandom
	b.wantRandom = false
	b.mu.Unlock()
	if wantRandom {
		b.fetchRandomFrom(entries)
	}
}

func (b *PolyhavenBrowser) failIndex(message string) {
	log.Printf("polyhaven: index %s", message)
	b.mu.Lock()
	b.status = polyhavenFailed
	b.mu.Unlock()
}

func (b *PolyhavenBrowser) fetchModel(entry PolyhavenEntry) {
	data, ok := httpGetBytes(polyhavenFilesURL + entry.Slug)
	if !ok {
		b.clearLoading()
		return
	}
	var files polyhavenModelFiles
	if err := json.Unmarshal(data, &files); err != nil {
		b.clearLoading()
		return
	}
	gltf, ok := pickPolyhavenResolution(files.Gltf, polyhavenPreferredResolution)
	if !ok {
		b.clearLoading()
		return
	}
	gltfBytes, ok := httpGetBytes(gltf.URL)
	if !ok {
		b.clearLoading()
		return
	}
	resources := make(map[string][]byte, len(gltf.Include))
	for relativePath, file := range gltf.Include {
		resourceBytes, got := httpGetBytes(file.URL)
		if !got {
			b.clearLoading()
			return
		}
		resources[relativePath] = resourceBytes
	}
	b.publishPending(&PendingModel{
		DisplayName: entry.Name,
		GltfBytes:   gltfBytes,
		Resources:   resources,
	})
}

func pickPolyhavenResolution(resolutions map[string]polyhavenGltfResolution, preferred uint32) (polyhavenGltfFile, bool) {
	if len(resolutions) == 0 {
		return polyhavenGltfFile{}, false
	}
	type candidate struct {
		value uint32
		gltf  polyhavenGltfFile
	}
	candidates := make([]candidate, 0, len(resolutions))
	for key, resolution := range resolutions {
		candidates = append(candidates, candidate{value: polyhavenResolutionValue(key), gltf: resolution.Gltf})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].value < candidates[j].value })
	for _, item := range candidates {
		if item.value == preferred {
			return item.gltf, true
		}
	}
	chosen := candidates[0]
	for _, item := range candidates {
		if item.value <= preferred {
			chosen = item
		}
	}
	return chosen.gltf, true
}

func polyhavenResolutionValue(key string) uint32 {
	trimmed := strings.TrimRight(key, "kK")
	if value, err := strconv.ParseUint(trimmed, 10, 32); err == nil {
		return uint32(value)
	}
	return ^uint32(0)
}

func polyhavenEntriesFromRaw(raw map[string]polyhavenRawAsset) []PolyhavenEntry {
	entries := make([]PolyhavenEntry, 0, len(raw))
	for slug, asset := range raw {
		entries = append(entries, PolyhavenEntry{Slug: slug, Name: asset.Name})
	}
	sort.Slice(entries, func(i, j int) bool {
		return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name)
	})
	return entries
}
