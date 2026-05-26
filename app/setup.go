package app

import (
	"fmt"

	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/ui"
	"github.com/matthewjberger/indigo/window"
)

type EngineRef struct{ World *ecs.World }

func NewEngineWorld(renderer *render.Renderer) (*ecs.World, error) {
	engine := ecs.New()
	ecs.Register[transform.LocalTransform](engine)
	ecs.Register[transform.GlobalTransform](engine)
	ecs.Register[transform.Parent](engine)
	ecs.Register[transform.LocalTransformDirty](engine)
	ecs.Register[transform.IgnoreParentScale](engine)
	ecs.Register[transform.GroupRoot](engine)
	ecs.Register[asset.RenderMesh](engine)
	ecs.Register[asset.SkinnedMesh](engine)
	ecs.Register[asset.Material](engine)
	ecs.Register[asset.MaterialVariants](engine)
	ecs.Register[asset.LodChain](engine)
	ecs.Register[asset.MorphWeights](engine)
	ecs.Register[asset.AnimationPlayer](engine)
	ecs.Register[render.Light](engine)
	ecs.Register[render.ParticleEmitter](engine)
	ecs.Register[render.CameraMarker](engine)
	ecs.Register[render.PickProxy](engine)
	ecs.Register[render.Selected](engine)
	ecs.Register[Name](engine)

	meshAssets := asset.NewMeshAssets()
	skinnedMeshAssets := asset.NewSkinnedMeshAssets()
	primitives, err := asset.RegisterPrimitives(renderer.Device, meshAssets)
	if err != nil {
		return nil, fmt.Errorf("app: register primitives: %w", err)
	}

	textureCache := asset.NewTextureCache()
	if _, err := textureCache.EnsureWhite(renderer.Device, renderer.Queue); err != nil {
		return nil, fmt.Errorf("app: register white texture: %w", err)
	}

	materialArrays, err := asset.NewMaterialTextureArrays(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: material texture arrays: %w", err)
	}

	materialRegistry, err := asset.NewMaterialRegistry(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: material registry: %w", err)
	}

	ibl, err := pass.NewIBL(renderer.Device, renderer.Queue)
	if err != nil {
		return nil, fmt.Errorf("app: ibl: %w", err)
	}

	shadow, err := pass.NewShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: shadow: %w", err)
	}

	spotShadow, err := pass.NewSpotShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: spot shadow: %w", err)
	}

	pointShadow, err := pass.NewPointShadow(renderer.Device)
	if err != nil {
		return nil, fmt.Errorf("app: point shadow: %w", err)
	}

	lines := &pass.Lines{}

	ecs.SetResource(engine, window.Window{
		Viewport: window.ViewportSize{
			Width:  renderer.Config.Width,
			Height: renderer.Config.Height,
		},
	})
	ecs.SetResource(engine, render.RendererResource{Renderer: renderer})
	ecs.SetResource(engine, asset.MeshAssetsResource{Assets: meshAssets})
	ecs.SetResource(engine, asset.SkinnedMeshAssetsResource{Assets: skinnedMeshAssets})
	ecs.SetResource(engine, asset.TextureCacheResource{Cache: textureCache})
	ecs.SetResource(engine, asset.MaterialTextureArraysResource{Arrays: materialArrays})
	ecs.SetResource(engine, asset.LoadingQueueResource{Queue: asset.NewLoadingQueue(materialArrays)})
	ecs.SetResource(engine, asset.MaterialRegistryResource{Registry: materialRegistry})
	ecs.SetResource(engine, pass.IBLResource{IBL: ibl})
	ecs.SetResource(engine, pass.ShadowResource{Shadow: shadow})
	ecs.SetResource(engine, pass.SpotShadowResource{Shadow: spotShadow})
	ecs.SetResource(engine, pass.PointShadowResource{Shadow: pointShadow})
	ecs.SetResource(engine, pass.LinesResource{Lines: lines})
	ecs.SetResource(engine, primitives)
	ecs.SetResource(engine, render.NewInput())
	ecs.SetResource(engine, render.DefaultGraphics())
	ecs.SetResource(engine, render.DefaultEngineConfig())
	ecs.SetResource(engine, transform.NewState())
	ecs.SetResource(engine, ecs.EcsCommandQueueResource{})
	ecs.SetResource(engine, render.CommandQueueResource{})
	return engine, nil
}

type Worlds struct {
	Engine *ecs.World
	Game   *ecs.World
	UI     *ecs.World

	EngineSchedule *ecs.Schedule
	GameSchedule   *ecs.Schedule
	UISchedule     *ecs.Schedule
}

func NewWorlds(renderer *render.Renderer) (Worlds, error) {
	engine, err := NewEngineWorld(renderer)
	if err != nil {
		return Worlds{}, err
	}

	game := ecs.New()
	ecs.Register[EngineEntity](game)
	ecs.SetResource(game, window.Window{})
	ecs.SetResource(game, EngineRef{World: engine})

	uiWorld := ui.NewWorld(renderer.Config.Width, renderer.Config.Height)
	ecs.SetResource(engine, ui.WorldRef{World: uiWorld})

	return Worlds{
		Engine:         engine,
		Game:           game,
		UI:             uiWorld,
		EngineSchedule: ecs.NewSchedule(),
		GameSchedule:   ecs.NewSchedule(),
		UISchedule:     ui.NewSchedule(),
	}, nil
}

func TickFrame(worlds Worlds, hooks *App, delta float32) {
	window.Advance(&ecs.MustResource[window.Window](worlds.Engine).Timing, delta)
	window.Advance(&ecs.MustResource[window.Window](worlds.Game).Timing, delta)
	if worlds.UI != nil {
		window.Advance(&ecs.MustResource[window.Window](worlds.UI).Timing, delta)
	}

	if worlds.UI != nil && worlds.UISchedule != nil {
		worlds.UISchedule.Run(worlds.UI)
	}
	worlds.GameSchedule.Run(worlds.Game)
	worlds.EngineSchedule.Run(worlds.Engine)

	if hooks != nil {
		if hooks.RunSystems != nil {
			hooks.RunSystems(worlds.Engine)
		}
		if hooks.PreRender != nil {
			hooks.PreRender(worlds.Engine)
		}
	}

	ecs.ProcessEcsCommands(worlds.Engine)
	worlds.Engine.ApplyCommands()
	worlds.Game.ApplyCommands()
	if worlds.UI != nil {
		worlds.UI.ApplyCommands()
	}
}

func PostFrame(worlds Worlds) {
	worlds.Engine.Step()
	worlds.Game.Step()
	if worlds.UI != nil {
		worlds.UI.Step()
	}
	render.InputBeginFrame(ecs.MustResource[render.Input](worlds.Engine))
}
