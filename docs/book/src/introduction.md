# Introduction

indigo is a data-oriented 3D game engine written in Go. It handles rendering, retained UI, input, transform hierarchy, and asset loading, and exposes them through plain structs of function values that an application fills in. The engine is the kernel that runs every frame. The application is the data on the worlds and the per-frame logic that runs alongside it.

The runtime is built on four pieces. A custom archetype Entity Component System gives the engine its storage and iteration model. A wgpu-backed renderer drives a declarative render graph with clustered lighting and image-based PBR. GLFW handles desktop windowing on Windows, macOS, and Linux; the same binaries target WebAssembly + canvas + WebGPU in the browser. glTF is the model format, loaded into typed caches and spawned as entities with parent links into the transform hierarchy.

## How an indigo application is shaped

An application is an `app.App` struct: a bundle of optional lifecycle hooks the main loop invokes at fixed points each frame.

```go
type App struct {
    Initialize           func(world *ecs.World)
    ConfigureRenderGraph func(world *ecs.World, renderer *render.Renderer)
    RunSystems           func(world *ecs.World)
    PreRender            func(world *ecs.World)
}
```

`Initialize` runs once after the renderer is built and sets up the scene, the camera, the lights, and any entities the app needs. `ConfigureRenderGraph` runs once after that to register passes against the engine's resources. `RunSystems` runs every frame before rendering and is where game logic lives. Every hook is optional; nil fields are skipped.

The engine ships with three worlds rather than one. The **engine world** holds rendering data — transforms, mesh handles, materials, lights, the camera, input, the renderer handle, graphics settings, the window's viewport and timing. The **game world** holds gameplay data — Breakout's paddle, ball, bricks; the editor's spinner system; whatever the application puts there. The **UI world** is a third instance reserved for retained-UI state: nodes, colors, text, parents, hit-test rects, focus, text-input buffers.

```
┌────────────────────────────────────────────────────────────────┐
│                       Your App (app.App)                        │
├────────────────────────────────────────────────────────────────┤
│  Initialize │ ConfigureRenderGraph │ RunSystems │ PreRender    │
└─────────────┴──────────────────────┴────────────┴──────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Engine World │      │  Game World  │      │   UI World   │
│  rendering   │◀────▶│  gameplay    │      │  retained UI │
└──────────────┘ sync └──────────────┘      └──────────────┘
       │                                            ▲
       └────────────────────────────────────────────┘
                          ui.WorldRef
```

Cross-world reach is explicit. `app.EngineRef` lives on the game world and points sync helpers (`app.SyncEngineTransform` and friends) at the engine world. `ui.WorldRef` lives on the engine world and points the renderer's UI passes at the UI world. `app.NewWorlds(renderer)` constructs all three together and installs both bridges, so a hand-built app cannot silently forget one.

## The frame loop

Each frame proceeds in a fixed order, written out as plain Go in the platform layer. There is no `engine.Run()` that takes control.

```
poll window events
compute delta
sync UI pointer state from the raw input snapshot

TickFrame(worlds, app, delta)
    advance window timing on each world
    UI schedule, game schedule, engine schedule
    app.RunSystems, app.PreRender
    apply deferred ECS commands

handle right-click, drive text inputs, handle UI clicks

render.RenderFrame(renderer, engine)
    execute compiled graph passes in topological order
    present

PostFrame(worlds)
    Step on each world (rolls events + change-detection watermarks)
    Input.BeginFrame (clears per-frame deltas)
```

Schedules are named slices of `func(*ecs.World)` values. Inter-system data flows through the world's resources rather than through arguments, which keeps every system's signature identical and trivial to test in isolation.

## What the engine ships with

Rendering is image-based PBR with a metallic-roughness workflow. Directional, point, and spot lights are culled per-frame into a 16×9×24 cluster grid by two compute dispatches the mesh pass owns. A procedural sky cubemap feeds an IBL filter; the filtered cubemap, BRDF LUT, and material textures land on the mesh pass. The postprocess pass does an ACES filmic tonemap from the HDR scene buffer into an LDR target; FXAA, UI, and the final present pass build on that.

The render graph is a DAG of passes that declare their reads and writes. Compile topologically sorts the passes, allocates any transient textures whose handles are still empty, decides clear-vs-load by who writes first, and freezes an execution order. RenderFrame walks the sorted list every frame.

Retained UI is its own ECS world. The layout system and the interaction system are systems on that world; the renderer's UI quad and UI text passes iterate its components the same way mesh passes iterate the engine world's. The bitmap font atlas is hand-rolled and owned directly by the UI text pass — no third-party font dependency to ship with wasm builds.

Input covers keyboard and mouse through GLFW on the desktop; the same `render.Input` struct is filled from JavaScript event listeners in the browser. Picking goes through a dedicated render pass that draws entity IDs into an offscreen buffer and reads them back asynchronously. A handful of debug overlays — bounding volumes, normals, the editor grid — go through the lines pass. Gizmos and selection outlines are their own passes overlaid on the LDR scene before present.

The same code runs on Windows, Linux, macOS, and the web. The desktop builds are plain `go build`. The web build is `GOOS=js GOARCH=wasm go build` plus `wasm_exec.js`.

## What ships in this repository

Two end-to-end consumers ship alongside the engine. The **editor** under `cmd/editor` loads glTF scenes, drives the pan/orbit camera, picks entities, and demonstrates the full render graph. The **breakout** demo under `cmd/breakout` shows the smaller surface area — a game world that drives the engine world through `app.Sync*` helpers, retained UI for the HUD, and a custom render graph configuration.

## What this book covers

The next chapters walk through installation, the first application, the project layout, and the architecture in detail. The ECS, the render graph, the asset pipeline, and the retained UI each get their own section. The two example applications get walk-throughs at the end.

Architecture notes also live at [docs/ARCHITECTURE.md](https://github.com/matthewjberger/indigo/blob/main/docs/ARCHITECTURE.md) inside the repository. The source is at [github.com/matthewjberger/indigo](https://github.com/matthewjberger/indigo). Live demos are at [matthewberger.dev/indigo/editor/](https://matthewberger.dev/indigo/editor/) and [matthewberger.dev/indigo/breakout/](https://matthewberger.dev/indigo/breakout/).
