package pass

import (
	"fmt"
	"log"

	"indigo/ecs"
	"indigo/render"
)

// RebuildIbl is a render-queue command that re-runs the IBL bake
// pipeline (procedural sky capture, cubemap mips, Lambertian +
// GGX environment filter) on the engine-world's [IBL] resource at
// the start of the next frame.
//
// Use case: scene transitions, atmosphere swaps, day-night
// updates — anywhere the procedural sky shader's output should
// re-feed the irradiance / prefiltered cubemaps without
// restarting the engine.
//
// Limitation today: the mesh pass's IBL bind group references the
// old irradiance + prefiltered TextureView handles, which are
// released during [IBL.Rebake]. Callers must rebuild that bind
// group separately — wiring this through [render.Pass.InvalidateBindGroups]
// is the next step once a real consumer queues a Rebake.
type RebuildIbl struct{}

// Apply runs Rebake on the IBLResource and logs progress. Errors
// are logged rather than returned because the render-command
// drain path has no error sink.
func (RebuildIbl) Apply(world *ecs.World, renderer *render.Renderer) {
	resource, ok := ecs.Resource[IBLResource](world)
	if !ok {
		log.Printf("rebuild_ibl: no IBLResource on world")
		return
	}
	if err := resource.IBL.Rebake(renderer.Device, renderer.Queue); err != nil {
		log.Printf("rebuild_ibl: %v", fmt.Errorf("rebake: %w", err))
	}
}
