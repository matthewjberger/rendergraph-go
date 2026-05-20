package ui

import (
	"indigo/ecs"
	"indigo/window"
)

// NewWorld returns a fresh ECS world configured for UI: every UI
// component is registered, the layout scratch and pointer state are
// installed as resources, and the window resource is initialized
// with the supplied viewport size so the layout system has something
// to anchor against on the first frame.
//
// Layout policy: this is the only world the [LayoutSystem] /
// [InteractionSystem] systems operate on.
func NewWorld(viewportWidth, viewportHeight uint32) *ecs.World {
	world := ecs.New()
	Register(world)
	ecs.SetResource(world, window.Window{
		Viewport: window.ViewportSize{Width: viewportWidth, Height: viewportHeight},
	})
	ecs.SetResource(world, NewLayoutState())
	ecs.SetResource(world, PointerState{})
	return world
}

// NewSchedule returns the standard UI schedule with layout before
// interaction so the hit test queries this frame's Resolved rects.
func NewSchedule() *ecs.Schedule {
	sched := ecs.NewSchedule()
	sched.Push("ui_layout", LayoutSystem)
	sched.Push("ui_interaction", InteractionSystem)
	return sched
}
