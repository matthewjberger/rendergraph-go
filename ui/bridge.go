package ui

import "indigo/ecs"

// WorldRef is the engine-world resource that points at the UI
// world. The UI render passes install this on the engine world at
// startup so their Prepare/Execute callbacks can look up the UI
// world (which holds the [Node] / [Color] / [Text] components) from
// the standard render-graph PassContext, which only carries the
// engine world.
//
// Apps install one of these themselves when wiring the engine + UI
// worlds together; see [indigo/pass.AddUiQuadPass] for the
// expected setup.
type WorldRef struct {
	World *ecs.World
}

// HasUI reports whether engine carries a [WorldRef]. UI passes use
// this to no-op cleanly when an app forgets to wire the UI world.
func HasUI(engine *ecs.World) bool {
	return ecs.HasResource[WorldRef](engine)
}
