// Package ui is indigo's retained-UI substrate.
//
// UI lives in its own ECS world (parallel to the engine and game
// worlds; see [indigo/app.Worlds]). UI entities carry a small set of
// data-only components: [Node] for the rectangle, [Color] for fill,
// [Text] for an optional label, [Parent] for tree structure,
// [Layout] for child flow rules, and [Interactive] for pointer
// state. A few free-function systems
// ([LayoutSystem], [InteractionSystem]) operate on those components
// every frame; the engine's render graph consumes the resulting
// laid-out tree through [indigo/pass.AddUiQuadPass] +
// [indigo/pass.AddUiTextPass].
//
// The package deliberately does not generate widget code (no derive
// macros, no per-widget structs the user must subclass). Apps build
// trees imperatively with [Builder].
package ui

import (
	"indigo/ecs"
)

// Anchor describes how a Node's position is interpreted relative to
// the screen. Anchor=TopLeft means (X, Y) is the pixel offset from
// the screen's top-left corner; Anchor=BottomRight means (X, Y) is
// the pixel offset from the bottom-right; etc. Parent-relative
// positioning is automatic when the entity has a [Parent] pointing
// at another UI node.
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

// LayoutMode controls how a Node arranges its children. None leaves
// children at the (X, Y) the builder set; Row stacks children left
// to right; Column stacks them top to bottom.
type LayoutMode uint8

const (
	LayoutNone LayoutMode = iota
	LayoutRow
	LayoutColumn
)

// Node is the rectangle component every UI entity carries.
// Resolved is the final screen-space rect filled by [LayoutSystem].
// Higher [ZIndex] draws on top. [Clip], when non-zero, culls the
// entity from render passes if its Resolved doesn't intersect.
// [Grow] is the flex weight used to share unused axis space among
// LayoutRow / LayoutColumn children.
type Node struct {
	X, Y          float32
	Width, Height float32
	Anchor        Anchor
	Padding       float32
	Spacing       float32
	Layout        LayoutMode
	Grow          float32
	ZIndex        int32
	Clip          Rect
	Resolved      Rect
	Dirty         bool
}

// Rect is a screen-space rectangle in pixels.
type Rect struct {
	X, Y          float32
	Width, Height float32
}

// Contains reports whether (px, py) lies inside r.
func (r Rect) Contains(px, py float32) bool {
	return px >= r.X && py >= r.Y && px < r.X+r.Width && py < r.Y+r.Height
}

// Color is the fill color for a Node. Stored as linear RGBA with
// alpha in [0,1]. The UI quad pass alpha-blends each colored node
// over whatever's underneath in scene_color.
type Color struct {
	RGBA [4]float32
}

// Text is an optional label attached to a UI entity. Rendered using
// indigo's hand-rolled bitmap font; missing glyphs are drawn as a
// blank cell. Position is centered inside the Node's Resolved rect.
type Text struct {
	Content string
	Color   [4]float32
	Scale   float32
}

// Parent links a UI entity to another UI entity that should contain
// it. Layout systems read this to do parent-relative positioning and
// flow.
type Parent struct {
	Entity ecs.Entity
}

// Interactive marks a UI entity as pointer-reactive. The
// interaction system updates Hovered and Pressed each frame and
// raises an EntityClicked event on the press-release cycle.
type Interactive struct {
	Hovered bool
	Pressed bool
}

// EntityClicked is the event the interaction system emits when an
// Interactive entity sees a press-release with the pointer still
// over it. Consume via [ecs.ReadEvents] or [ecs.DrainEvents] on the
// UI world.
type EntityClicked struct {
	Entity ecs.Entity
}

// TextInput is the editable-text component. When the entity is
// focused, [AdvanceTextInputs] appends typed runes to Buffer and
// updates the entity's Text.Content for display.
type TextInput struct {
	Buffer string
	Caret  int
}

// TextCommitted is emitted on Enter; consumers write Value back to
// their model.
type TextCommitted struct {
	Entity ecs.Entity
	Value  string
}

// Register installs every UI component on world. Call once per UI
// world (NewWorld handles this for the standard case).
func Register(world *ecs.World) {
	ecs.Register[Node](world)
	ecs.Register[Color](world)
	ecs.Register[Text](world)
	ecs.Register[Parent](world)
	ecs.Register[Interactive](world)
	ecs.Register[TextInput](world)
}
