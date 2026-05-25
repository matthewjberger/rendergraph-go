package asset

import "github.com/cogentcore/webgpu/wgpu"

// LoadingQueue streams glTF texture decoding and GPU upload across frames so
// geometry can be displayed immediately while textures fill in behind a
// placeholder. Reserved array layers are stable, so decoded pixels upload in
// place without any material rewrite.
//
// Image decoding (the dominant cost) runs through a platform-specific
// textureDecoder: on native it fans out to a goroutine worker pool for real
// parallelism, on wasm it decodes inline within the per-frame budget. GPU
// uploads always happen on the main thread inside Drain.
type LoadingQueue struct {
	arrays     *MaterialTextureArrays
	decoder    *textureDecoder
	generation uint64
	total      int
	completed  int
	label      string
}

type LoadingQueueResource struct {
	Queue *LoadingQueue
}

type pendingTexture struct {
	layer      uint32
	srgb       bool
	encoded    []byte
	generation uint64
}

type decodedLayer struct {
	layer      uint32
	srgb       bool
	rgba       []byte
	generation uint64
	err        error
}

// NewLoadingQueue creates a streaming queue bound to the given texture arrays
// and registers itself so MaterialTextureArrays.Reset clears it too.
func NewLoadingQueue(arrays *MaterialTextureArrays) *LoadingQueue {
	queue := &LoadingQueue{arrays: arrays, decoder: newTextureDecoder()}
	arrays.loading = queue
	return queue
}

// SetLabel records the name of the asset currently streaming, surfaced by the
// progress UI.
func (q *LoadingQueue) SetLabel(label string) {
	q.label = label
}

// Label returns the name of the asset currently streaming.
func (q *LoadingQueue) Label() string {
	return q.label
}

// enqueue submits the encoded bytes of an already-reserved layer for streamed
// decode and upload.
func (q *LoadingQueue) enqueue(layer uint32, srgb bool, encoded []byte) {
	q.total++
	q.decoder.submit(pendingTexture{
		layer:      layer,
		srgb:       srgb,
		encoded:    encoded,
		generation: q.generation,
	}, q.arrays.LayerSize)
}

// Drain uploads up to the per-frame budget of decoded textures into their
// reserved layers. Results from a superseded scene (older generation) are
// discarded without counting toward progress.
func (q *LoadingQueue) Drain(queue *wgpu.Queue) {
	for processed := 0; processed < streamBudget; {
		result, ok := q.decoder.poll()
		if !ok {
			break
		}
		if result.generation != q.generation {
			continue
		}
		q.completed++
		processed++
		if result.err != nil {
			continue
		}
		_ = q.arrays.UploadLayer(queue, result.layer, result.srgb, result.rgba, q.arrays.LayerSize, q.arrays.LayerSize)
	}
}

// Reset abandons any in-flight work for the current scene. In-flight decodes
// already dispatched complete but are discarded by the generation check.
func (q *LoadingQueue) Reset() {
	q.generation++
	q.total = 0
	q.completed = 0
	q.decoder.reset()
}

// Active reports whether textures are still streaming.
func (q *LoadingQueue) Active() bool {
	return q.completed < q.total
}

// Progress returns the number of completed and total streamed textures.
func (q *LoadingQueue) Progress() (completed, total int) {
	return q.completed, q.total
}

// Fraction returns streaming progress in [0,1]; 1 when idle.
func (q *LoadingQueue) Fraction() float32 {
	if q.total == 0 {
		return 1
	}
	return float32(q.completed) / float32(q.total)
}

func decodeLayerJob(item pendingTexture, layerSize uint32) decodedLayer {
	result := decodedLayer{layer: item.layer, srgb: item.srgb, generation: item.generation}
	pixels, width, height, err := decodeImageBytes(item.encoded)
	if err != nil {
		result.err = err
		return result
	}
	result.rgba = resizeRGBA(pixels, width, height, layerSize, layerSize)
	return result
}
