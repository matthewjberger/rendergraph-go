package pass

import (
	"fmt"
	"log"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
)

type RebuildIbl struct{}

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
