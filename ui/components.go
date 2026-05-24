package ui

import (
	"github.com/matthewjberger/indigo/ecs"
)

type Anchor uint8

const (
	AnchorTopLeft Anchor = iota
	AnchorTopRight
	AnchorBottomLeft
	AnchorBottomRight
	AnchorCenter
	AnchorTopCenter
	AnchorBottomCenter
)

type LayoutMode uint8

const (
	LayoutNone LayoutMode = iota
	LayoutRow
	LayoutColumn
)

type Node struct {
	X, Y           float32
	Width, Height  float32
	Anchor         Anchor
	Padding        float32
	Spacing        float32
	Layout         LayoutMode
	Grow           float32
	ZIndex         int32
	Hidden         bool
	ClipChildren   bool
	Resolved       Rect
	ClipResolved   Rect
	HiddenResolved bool
	Dirty          bool
}

type Rect struct {
	X, Y          float32
	Width, Height float32
}

func (r Rect) Contains(px, py float32) bool {
	return px >= r.X && py >= r.Y && px < r.X+r.Width && py < r.Y+r.Height
}

type Color struct {
	RGBA [4]float32
}

type Text struct {
	Content string
	Color   [4]float32
	Scale   float32
}

type Parent struct {
	Entity ecs.Entity
}

type Interactive struct {
	Hovered bool
	Pressed bool
}

type EntityClicked struct {
	Entity ecs.Entity
}

type TextInput struct {
	Buffer string
	Caret  int
}

type TextCommitted struct {
	Entity ecs.Entity
	Value  string
}

// SetVisible shows or hides a node and its whole subtree. It marks layout dirty
func SetVisible(world *ecs.World, entity ecs.Entity, visible bool) {
	node, ok := ecs.GetMut[Node](world, entity)
	if !ok {
		return
	}
	if node.Hidden == !visible {
		return
	}
	node.Hidden = !visible
	MarkLayoutDirty(world)
}

func Register(world *ecs.World) {
	ecs.Register[Node](world)
	ecs.Register[Color](world)
	ecs.Register[Text](world)
	ecs.Register[Parent](world)
	ecs.Register[Interactive](world)
	ecs.Register[TextInput](world)
}
