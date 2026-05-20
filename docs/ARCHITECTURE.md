# Architecture

rendergraph-go is a data-oriented Go game engine ported from
[nightshade](https://github.com/matthewjberger/nightshade), built on
[freecs-go](https://github.com/matthewjberger/freecs-go) (vendored
inline as `ecs/`).

## Principles

- **Data-oriented, not OO.** Data and free functions, never objects
  with methods that own business logic. Where nightshade's Rust uses
  traits (`State`, `PassNode`), rendergraph-go uses structs whose
  fields are function values: `Pass{Reads, Writes, State,
  Prepare, Execute, InvalidateBindGroups, Release}` and
  `App{Initialize, ConfigureRenderGraph, RunSystems, PreRender}`.
- **Dual ECS worlds.** An engine world holds rendering data
  (transforms, RenderMesh, camera, input, graphics settings, lights,
  window timing). A game world holds game state (e.g. spinners),
  linked to the engine world via an `app.EngineEntity{Entity}` bridge
  component. Game systems write back into the engine world only
  through named `app.SyncEngine...` helpers.
- **Render graph for rendering.** Passes declare which slot names
  they read and write. The graph wires slots to resources, topo-sorts
  passes by their read/write dependencies, decides clear-vs-load
  based on first-write, and invalidates bind groups when read-slot
  resource versions change.
- **Free-function systems, ordered by named schedule.** Engine and
  game each have an `ecs.Schedule` of named systems with the
  canonical `func(*World)` signature. Inter-system data flows through
  ECS resources, not through arguments.

## Module layout

| Package    | Purpose                                                |
|------------|--------------------------------------------------------|
| `ecs/`     | freecs-go vendored inline (archetype ECS, events, schedule). |
| `transform/` | `LocalTransform`, `GlobalTransform`, `Parent`, dirty propagation. |
| `window/`  | `Window{Viewport, Timing}` resource, `Advance(timing, delta)`. |
| `render/`  | Renderer, render graph, passes (mesh, grid, sky, FXAA, present), camera, input, lights, graphics settings. |
| `app/`     | `App` lifecycle hooks, `EngineEntity` bridge component, `SyncEngine*` helpers. |
| `cmd/rendergraph-go/` | Demo entry points (desktop `!js`, wasm `js`), schedule wiring, spinner system. |
| `cmd/serve/` | Tiny file server for the wasm bundle. |

No package depends on a higher layer. `render` depends on
`ecs`+`transform`+`window`; `app` depends on `ecs`+`render`+`transform`;
`cmd/rendergraph-go` ties everything together.

## ECS

The `ecs` package is freecs-go renamed. Conventions:

- Components are plain structs, registered with
  `ecs.Register[MyComponent](world)`. Up to 64 component types per
  world (so we use two worlds).
- Resources are keyed by Go type. Named types
  (`type DeltaTime float32`) give each scalar its own slot.
- Systems are free functions with signature `func(*ecs.World)`.
  Push them onto an `ecs.Schedule` and call `schedule.Run(world)`.
- Hot iteration uses `world.ForEach(mask, exclude, callback)` with
  `ecs.Column[T](world, table)` for direct typed column access.
- `ecs.EntityDespawned` is a built-in event emitted from `Despawn`;
  the mesh pass consumes it to release per-entity slot allocations
  in O(despawned) rather than scanning every slot.

## Dual-world bridge

Game-side entities link to engine-side entities via
`app.EngineEntity{Entity}`. The spinner demo is the live example:
each spinner game entity carries an `EngineEntity` pointing at its
mesh-rendering twin in the engine world. `advanceSpinners` walks the
game world's `(Spinner | EngineEntity)` archetype with `ForEach`,
updates the game-side rotation, then writes the result through
`app.SyncEngineRotation` (which mutates the engine
`LocalTransform` and marks it dirty). Direct cross-world mutation is
convention-only; the discipline is "go through `SyncEngine*` or
nothing."

## Render graph

The Go render graph keeps the fundamentals of nightshade's:

- `ResourceID` handles into a flat descriptor / handle / version
  table.
- Resource kinds: external / transient × color / depth. External
  resources are refreshed by the caller each frame (the swapchain
  view); transients are owned by the graph and (re)allocated on
  resize.
- `Pass` is a struct of function-value fields plus declared
  `Reads` / `Writes` slot lists and opaque `State`. The graph never
  dispatches through an interface; the `State any` field is cast at
  the top of each pass function.
- `Compile(device)` builds a dependency DAG from each pass's Reads
  and Writes (readers depend on the latest writer of their resource;
  write-after-write preserves insertion order), topologically sorts
  with Kahn's algorithm (insertion-order tiebreak for stability),
  records the first-write clear-ops, and auto-allocates any
  transients whose handles are still empty.
- `Execute` runs every pass in topo order. Before each pass it
  compares only the pass's *read*-slot resource versions against a
  per-pass snapshot; on any mismatch it calls
  `Pass.InvalidateBindGroups` so cached bind groups referencing
  stale views get rebuilt. Write-only attachments do not trigger
  invalidation.
- `PassContext` holds device, queue, encoder, the ECS world, the
  resource table, this pass's slot bindings, and the clear-ops
  set. Helpers return fully-prepared
  `wgpu.RenderPassColorAttachment` / `RenderPassDepthStencilAttachment`
  structs ready to drop into a render-pass descriptor.

### Default graph layout

`render.NewRenderer` registers three resources: a transient
`scene_color` (surface format, render-attachment + texture-binding +
copy-src), a transient `depth`, and an external `swapchain` that the
caller refreshes each frame. Three mesh primitives (`UnitTriangle`,
`UnitQuad`, `UnitCube`) are registered in the mesh-asset registry
and exposed via `Renderer.UnitTriangle` / `UnitQuad` / `UnitCube`.

The demo's `ConfigureRenderGraph` adds a fourth transient
(`fxaa_output`) and five passes; the topo sort orders them as:

```
sky ──► scene_color
mesh ──► scene_color, depth
grid ──► scene_color, depth          (loads, no clear)
fxaa: scene_color ──► fxaa_output
present: fxaa_output ──► swapchain
```

The sky pass is the first writer of scene_color so it gets the
clear-on-load op; the mesh and grid passes load and overwrite the
pixels they cover. The FXAA pass samples scene_color into
fxaa_output; the present pass blits fxaa_output to the swapchain.

### Adding a pass

```go
myPass, _ := render.NewMyPass(device, format)
renderer.Graph.AddPass(myPass, []render.SlotBinding{
    {Slot: "input",  ResourceID: someTransientID},
    {Slot: "output", ResourceID: anotherTransientID},
})
```

`Compile(device)` after the application's `ConfigureRenderGraph`
sorts everything and allocates any new transients.

### Deliberately deferred

- Transient aliasing pool. Each transient is its own texture today;
  nightshade reuses textures with compatible descriptors.
- Dead-pass culling.
- Subgraphs (nightshade uses these for per-camera viewport
  rendering).
- Buffer resources (only textures today).

## Shading

The mesh pass binds three bind groups:

- group 0: view-projection uniform
- group 1: per-handle storage buffer of model matrices (sparse-
  updated via `ecs.IterChanged1[transform.GlobalTransform]`, with
  stable per-entity slots)
- group 2: lights uniform (up to `render.MaxLights` packed
  `LightData` entries with vec3 alignment padding matching the WGSL
  struct)

The fragment shader derives a flat normal from `dpdx`/`dpdy` of the
world-space varying and applies Lambertian shading per light on top
of an ambient floor. Light direction comes from the entity transform's
-Z column (glTF KHR_lights_punctual convention, matching nightshade).

## Frame lifecycle

```
glfw window + wgpu instance + surface
└── render.NewRenderer
    ├── default graph (scene_color, depth, swapchain)
    └── primitive registry (unit_triangle, unit_quad, unit_cube)

ecs.New()  ×2  (engine + game)
└── register components, install resources
    (window, renderer, camera, input, pan-orbit controller,
     graphics settings, propagation state, mesh assets, ...)

app.ConfigureRenderGraph
└── add transients + passes, then graph.Compile(device)

each frame:
  glfw.PollEvents + delta computation
  window.Advance(timing, delta)   on both worlds
  gameSchedule.Run(game)          spinner system
  engineSchedule.Run(engine)      graphics_toggles, pan_orbit_camera,
                                  transform_propagation
  app.RunSystems / PreRender      (optional, both nil for the demo)
  world.ApplyCommands + Step      on both worlds
  Input.BeginFrame                clear per-frame deltas
  renderer.RenderFrame(engine)    set swapchain external, execute graph,
                                  present
```

## Why a struct of functions instead of a `PassNode` interface

In nightshade, `PassNode` is a trait with `&mut self` methods. In Go
the equivalent would be an interface with methods on a concrete
struct that owns the pass's GPU state. That works, but it conflates
the pass's data (pipeline, buffers, bind groups) with its behavior
(prepare + execute funcs). The data-oriented split lifts behavior to
function values and keeps state purely as fields. Passes compose by
swapping function pointers, not by overriding methods.
