package render

import (
	"fmt"
	"sort"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/matthewjberger/indigo/ecs"
)

type SlotBinding struct {
	Slot       string
	ResourceID ResourceID
}

type passEntry struct {
	pass         *Pass
	bindings     map[string]ResourceID
	lastVersions map[ResourceID]uint64
}

type Graph struct {
	Resources Resources

	passes         []passEntry
	executionOrder []int
	clearOps       map[ClearKey]struct{}
	compiled       bool
}

func NewGraph() *Graph {
	return &Graph{clearOps: make(map[ClearKey]struct{})}
}

func (g *Graph) ResourceByName(name string) ResourceID {
	for index := range g.Resources.Descriptors {
		if g.Resources.Descriptors[index].Name == name {
			return ResourceID(index)
		}
	}
	return 0
}

func (g *Graph) AddColorTexture(descriptor ResourceDescriptor) ResourceID {
	g.compiled = false
	return g.Resources.Register(descriptor)
}

func (g *Graph) AddDepthTexture(descriptor ResourceDescriptor) ResourceID {
	g.compiled = false
	return g.Resources.Register(descriptor)
}

func (g *Graph) AddPass(pass *Pass, bindings []SlotBinding) error {
	indexed := make(map[string]ResourceID, len(bindings))
	for _, binding := range bindings {
		indexed[binding.Slot] = binding.ResourceID
	}
	for _, slot := range pass.Reads {
		if _, ok := indexed[slot]; !ok {
			return fmt.Errorf("render: pass %q reads slot %q with no binding", pass.Name, slot)
		}
	}
	for _, slot := range pass.Writes {
		if _, ok := indexed[slot]; !ok {
			return fmt.Errorf("render: pass %q writes slot %q with no binding", pass.Name, slot)
		}
	}
	g.passes = append(g.passes, passEntry{
		pass:         pass,
		bindings:     indexed,
		lastVersions: make(map[ResourceID]uint64, len(pass.Reads)),
	})
	g.compiled = false
	return nil
}

func (g *Graph) Compile(device *wgpu.Device) error {
	order, err := g.topoSort()
	if err != nil {
		return err
	}
	g.executionOrder = order

	clearOps := make(map[ClearKey]struct{}, len(g.passes))
	seen := make(map[ResourceID]struct{}, len(g.Resources.Descriptors))
	for _, passIndex := range order {
		entry := g.passes[passIndex]
		for _, slot := range entry.pass.Writes {
			id := entry.bindings[slot]
			if _, already := seen[id]; already {
				continue
			}
			seen[id] = struct{}{}
			descriptor := g.Resources.Descriptor(id)
			if descriptor.ClearColor != nil || descriptor.ClearDepth != nil {
				clearOps[ClearKey{PassIndex: passIndex, ResourceID: id}] = struct{}{}
			}
		}
	}
	g.clearOps = clearOps

	if err := g.allocateMissingTransients(device); err != nil {
		return err
	}

	g.compiled = true
	return nil
}

func (g *Graph) topoSort() ([]int, error) {
	n := len(g.passes)
	edges := make([][]int, n)
	inDegree := make([]int, n)

	writers := make(map[ResourceID][]int, len(g.Resources.Descriptors))
	readers := make(map[ResourceID][]int, len(g.Resources.Descriptors))
	for i, entry := range g.passes {
		for _, slot := range entry.pass.Reads {
			id := entry.bindings[slot]
			readers[id] = append(readers[id], i)
		}
		for _, slot := range entry.pass.Writes {
			id := entry.bindings[slot]
			writers[id] = append(writers[id], i)
		}
	}

	addEdge := func(from, to int) {
		edges[from] = append(edges[from], to)
		inDegree[to]++
	}

	for _, ws := range writers {
		for k := 1; k < len(ws); k++ {
			addEdge(ws[k-1], ws[k])
		}
	}
	for id, rs := range readers {
		ws, ok := writers[id]
		if !ok || len(ws) == 0 {
			continue
		}
		lastWriter := ws[len(ws)-1]
		for _, r := range rs {
			if r == lastWriter {
				continue
			}
			addEdge(lastWriter, r)
		}
	}

	order := make([]int, 0, n)
	queue := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		sort.Ints(queue)
		node := queue[0]
		queue = queue[1:]
		order = append(order, node)
		for _, next := range edges[node] {
			inDegree[next]--
			if inDegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if len(order) != n {
		return nil, ErrGraphCycle
	}
	return order, nil
}

func (g *Graph) allocateMissingTransients(device *wgpu.Device) error {
	for index := range g.Resources.Descriptors {
		descriptor := &g.Resources.Descriptors[index]
		switch descriptor.Kind {
		case ResourceKindExternalColor, ResourceKindExternalDepth:
			continue
		case ResourceKindTransientColor, ResourceKindTransientDepth:
			if g.Resources.Handles[index].View != nil {
				continue
			}
			if err := g.createTransient(device, index); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *Graph) createTransient(device *wgpu.Device, index int) error {
	descriptor := &g.Resources.Descriptors[index]
	texture, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Label: descriptor.Name,
		Size: wgpu.Extent3D{
			Width:              descriptor.Texture.Width,
			Height:             descriptor.Texture.Height,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        descriptor.Texture.Format,
		Usage:         descriptor.Texture.Usage,
	})
	if err != nil {
		return fmt.Errorf("render: create transient texture %q: %w", descriptor.Name, err)
	}
	view, err := texture.CreateView(nil)
	if err != nil {
		texture.Release()
		return fmt.Errorf("render: create view for %q: %w", descriptor.Name, err)
	}
	g.Resources.Handles[index] = TextureHandle{
		Texture: texture,
		View:    view,
		Width:   descriptor.Texture.Width,
		Height:  descriptor.Texture.Height,
		Owned:   true,
	}
	g.Resources.Versions[index]++
	return nil
}

func (g *Graph) AllocateTransients(device *wgpu.Device) error {
	g.Resources.ReleaseOwned()
	return g.allocateMissingTransients(device)
}

func (g *Graph) ResizeTransients(device *wgpu.Device, width, height uint32) error {
	for index := range g.Resources.Descriptors {
		descriptor := &g.Resources.Descriptors[index]
		switch descriptor.Kind {
		case ResourceKindTransientColor, ResourceKindTransientDepth:
			descriptor.Texture.Width = width
			descriptor.Texture.Height = height
		}
	}
	return g.AllocateTransients(device)
}

func (g *Graph) Execute(device *wgpu.Device, queue *wgpu.Queue, world *ecs.World, encoder *wgpu.CommandEncoder) error {
	if !g.compiled {
		return ErrGraphNotCompiled
	}
	for _, passIndex := range g.executionOrder {
		entry := &g.passes[passIndex]
		dirty := false
		for _, slot := range entry.pass.Reads {
			id := entry.bindings[slot]
			if g.Resources.Versions[id] != entry.lastVersions[id] {
				dirty = true
				break
			}
		}
		if dirty && entry.pass.InvalidateBindGroups != nil {
			entry.pass.InvalidateBindGroups()
		}
		if dirty {
			for _, slot := range entry.pass.Reads {
				id := entry.bindings[slot]
				entry.lastVersions[id] = g.Resources.Versions[id]
			}
		}

		context := &PassContext{
			Device:    device,
			Queue:     queue,
			Encoder:   encoder,
			World:     world,
			Resources: &g.Resources,
			Slots:     entry.bindings,
			PassIndex: passIndex,
			ClearOps:  g.clearOps,
		}
		if entry.pass.Prepare != nil {
			if err := entry.pass.Prepare(context); err != nil {
				return fmt.Errorf("render: pass %q prepare: %w", entry.pass.Name, err)
			}
		}
		if entry.pass.Execute != nil {
			if err := entry.pass.Execute(context); err != nil {
				return fmt.Errorf("render: pass %q execute: %w", entry.pass.Name, err)
			}
		}
	}
	return nil
}

func (g *Graph) Release() {
	for _, entry := range g.passes {
		if entry.pass.Release != nil {
			entry.pass.Release()
		}
	}
	g.Resources.ReleaseOwned()
}
