package ui

import "github.com/matthewjberger/indigo/ecs"

// ScrollArea is a clipping viewport with a vertically scrollable content column.
type ScrollArea struct {
	Viewport ecs.Entity
	Content  ecs.Entity
	Offset   float32
}

// NewScrollArea builds a viewport of the given size that clips its descendants.
func NewScrollArea(b *Builder, width, height float32) ScrollArea {
	viewport := b.Node(Node{
		Width: width, Height: height,
		ClipChildren: true,
	}).Color(Color{RGBA: [4]float32{0, 0, 0, 0}}).Interactive().Entity()

	b.Push(viewport)
	content := b.Node(Node{
		Width:  width,
		Height: height,
		Layout: LayoutColumn,
	}).Entity()
	b.Pop()

	return ScrollArea{Viewport: viewport, Content: content}
}

// Update handles wheel scrolling while the pointer is over the viewport, clamps
func (s *ScrollArea) Update(world *ecs.World, pointerX, pointerY, wheel, contentHeight float32) bool {
	viewport, ok := ecs.Get[Node](world, s.Viewport)
	if !ok {
		return false
	}
	consumed := false
	if wheel != 0 && viewport.Resolved.Contains(pointerX, pointerY) {
		s.Offset -= wheel * 40
		consumed = true
	}
	maxOffset := contentHeight - viewport.Resolved.Height
	if maxOffset < 0 {
		maxOffset = 0
	}
	s.Offset = max(0, min(s.Offset, maxOffset))
	if content, ok := ecs.GetMut[Node](world, s.Content); ok && content.Y != -s.Offset {
		content.Y = -s.Offset
		MarkLayoutDirty(world)
	}
	return consumed
}
