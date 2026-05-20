package ui

import (
	"indigo/ecs"
	"indigo/window"
)

// LayoutSystem resolves every UI Node's screen-space rectangle each
// frame. Reads the viewport size from the UI world's [window.Window]
// resource, walks UI entities root-first, and writes the result back
// into Node.Resolved so downstream passes can position quads + text
// directly.
//
// Layout policy:
//   - A root Node (no UI [Parent]) is placed using its [Anchor] and
//     the (X, Y) offset from that anchor's reference corner.
//   - A child Node sits at (parent.Resolved.X + padding + X,
//     parent.Resolved.Y + padding + Y) by default.
//   - A parent with LayoutRow / LayoutColumn overrides the child's
//     intra-parent X/Y: children stack along the axis, separated by
//     parent.Spacing, with parent.Padding from the parent's edges.
//
// The walk is two passes: a depth pass that sorts entities by tree
// depth so parents resolve before children, then a placement pass.
// Depth is recomputed per frame since the tree is small and parent
// links can change.
func LayoutSystem(world *ecs.World) {
	state := ecs.Resource[LayoutState](world)
	state.Roots = state.Roots[:0]
	state.Order = state.Order[:0]
	for k := range state.Depths {
		delete(state.Depths, k)
	}
	for k := range state.Children {
		delete(state.Children, k)
	}

	nodeMask := ecs.MaskOf[Node](world)
	parentMask := ecs.MaskOf[Parent](world)

	world.ForEach(nodeMask, 0, func(entity ecs.Entity, _ *ecs.Archetype, _ int) {
		state.Order = append(state.Order, entity)
		if !world.HasComponents(entity, parentMask) {
			state.Roots = append(state.Roots, entity)
			state.Depths[entity] = 0
			return
		}
		parent, _ := ecs.Get[Parent](world, entity)
		state.Children[parent.Entity] = append(state.Children[parent.Entity], entity)
	})

	for _, root := range state.Roots {
		assignDepth(state, root, 0)
	}

	width, height := viewportSize(world)

	for _, entity := range state.Order {
		if _, isRoot := isRootCheck(world, entity, parentMask); isRoot {
			placeRoot(world, entity, width, height)
		}
	}
	for _, root := range state.Roots {
		placeChildren(world, state, root)
	}
}

// LayoutState is the per-frame layout scratch. Stored as a resource
// on the UI world so the system reuses its slices and maps.
type LayoutState struct {
	Roots    []ecs.Entity
	Order    []ecs.Entity
	Depths   map[ecs.Entity]int
	Children map[ecs.Entity][]ecs.Entity
}

// NewLayoutState returns an empty layout scratch with its maps
// allocated.
func NewLayoutState() LayoutState {
	return LayoutState{
		Depths:   make(map[ecs.Entity]int, 16),
		Children: make(map[ecs.Entity][]ecs.Entity, 16),
	}
}

func assignDepth(state *LayoutState, entity ecs.Entity, depth int) {
	state.Depths[entity] = depth
	for _, child := range state.Children[entity] {
		assignDepth(state, child, depth+1)
	}
}

func isRootCheck(world *ecs.World, entity ecs.Entity, parentMask ecs.Mask) (ecs.Entity, bool) {
	if !world.HasComponents(entity, parentMask) {
		return entity, true
	}
	return entity, false
}

func placeRoot(world *ecs.World, entity ecs.Entity, viewportW, viewportH float32) {
	node, _ := ecs.GetMut[Node](world, entity)
	x, y := anchoredOrigin(node, viewportW, viewportH)
	node.Resolved = Rect{X: x, Y: y, Width: node.Width, Height: node.Height}
}

func placeChildren(world *ecs.World, state *LayoutState, parent ecs.Entity) {
	children := state.Children[parent]
	if len(children) == 0 {
		return
	}
	parentNode, _ := ecs.Get[Node](world, parent)
	switch parentNode.Layout {
	case LayoutRow:
		cursor := parentNode.Resolved.X + parentNode.Padding
		y := parentNode.Resolved.Y + parentNode.Padding
		for _, child := range children {
			childNode, _ := ecs.GetMut[Node](world, child)
			childNode.Resolved = Rect{
				X:      cursor + childNode.X,
				Y:      y + childNode.Y,
				Width:  childNode.Width,
				Height: childNode.Height,
			}
			cursor += childNode.Width + parentNode.Spacing
		}
	case LayoutColumn:
		x := parentNode.Resolved.X + parentNode.Padding
		cursor := parentNode.Resolved.Y + parentNode.Padding
		for _, child := range children {
			childNode, _ := ecs.GetMut[Node](world, child)
			childNode.Resolved = Rect{
				X:      x + childNode.X,
				Y:      cursor + childNode.Y,
				Width:  childNode.Width,
				Height: childNode.Height,
			}
			cursor += childNode.Height + parentNode.Spacing
		}
	default:
		for _, child := range children {
			childNode, _ := ecs.GetMut[Node](world, child)
			childNode.Resolved = Rect{
				X:      parentNode.Resolved.X + parentNode.Padding + childNode.X,
				Y:      parentNode.Resolved.Y + parentNode.Padding + childNode.Y,
				Width:  childNode.Width,
				Height: childNode.Height,
			}
		}
	}

	for _, child := range children {
		placeChildren(world, state, child)
	}
}

func anchoredOrigin(node *Node, viewportW, viewportH float32) (float32, float32) {
	switch node.Anchor {
	case AnchorTopRight:
		return viewportW - node.Width - node.X, node.Y
	case AnchorBottomLeft:
		return node.X, viewportH - node.Height - node.Y
	case AnchorBottomRight:
		return viewportW - node.Width - node.X, viewportH - node.Height - node.Y
	case AnchorCenter:
		return (viewportW-node.Width)*0.5 + node.X, (viewportH-node.Height)*0.5 + node.Y
	case AnchorTopCenter:
		return (viewportW-node.Width)*0.5 + node.X, node.Y
	case AnchorBottomCenter:
		return (viewportW-node.Width)*0.5 + node.X, viewportH - node.Height - node.Y
	default:
		return node.X, node.Y
	}
}

func viewportSize(world *ecs.World) (float32, float32) {
	w := ecs.Resource[window.Window](world)
	return float32(w.Viewport.Width), float32(w.Viewport.Height)
}
