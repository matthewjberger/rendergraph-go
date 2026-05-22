# Architecture Overview

indigo is a layered dependency graph. Each layer builds on the one below it, and no layer references anything above it. This chapter walks the layers from the bottom up.

```
Application Layer:  cmd/editor, cmd/breakout, custom apps
                            |
App Layer:          app.App, app.Worlds, sync helpers, EngineRef
                            |
Feature Layer:      Retained UI, Picking, Gizmos, Outlines, Animation
                            |
Rendering Layer:    Render Graph → Passes → Materials → Textures → WGSL
                            |
Asset Layer:        Meshes, Textures, Materials, glTF, IBL
                            |
Core Layer:         Transform Hierarchy, Input, Window, Schedules
                            |
Foundation Layer:   ECS (archetype), Math (mgl32), GPU (wgpu)
```

## Foundation layer

The bottom of the graph is three independent systems that everything else depends on.

The ECS is a custom archetype Entity Component System under `ecs/`. Components are plain structs registered against a world. Entities are 64-bit generational handles. The storage lays out each component type as its own contiguous slice and groups entities into tables keyed by the exact set of component types they carry. Iteration walks the table directly through typed columns, which is a packed-memory loop the hardware prefetcher likes. Generics carry the typed accessors: `Get[T]`, `Set[T]`, `Resource[T]`, `Iter1..6[T1..T6]`. The `Iter` family is templated through a `go:generate` step rather than written by hand.

Math uses `github.com/go-gl/mathgl/mgl32` aliased through `transform/math.go` as `Vec2`, `Vec3`, `Mat4`, and `Quat`. `Mat4` is `[16]float32`, so transform code reads and writes packed column-major matrices without any extra indirection.

GPU access goes through a wgpu binding. One Go surface targets Vulkan, Metal, DirectX 12, and WebGPU. Every rendering pass in indigo is built against wgpu directly — there is no intermediate graphics abstraction.

## Core layer

Built on the foundation, the core layer manages per-frame state, the entity hierarchy, and the schedule mechanism.

The transform hierarchy propagates `LocalTransform` through `Parent` relationships to compute `GlobalTransform` matrices each frame. A `LocalTransformDirty` tag marks subtrees that need recomputation; `UpdateGlobalTransforms` walks the dirty set and propagates. `IgnoreParentScale` opts an entity out of inheriting its parent's scale, which the camera and certain UI nodes use.

Input aggregates keyboard, mouse, and scroll state each frame into the `render.Input` resource on the engine world. The desktop path reads from GLFW; the web path replays JavaScript events into the same struct. `InputBeginFrame` is called in `PostFrame` to clear per-frame deltas.

Windowing is `window.Window`, a resource on every world that carries the viewport size and a `Timing` struct. `window.Advance` advances the timing each frame; reading delta time is `MustResource[window.Window](world).Timing.DeltaSeconds`.

Schedules are `ecs.Schedule` values — ordered slices of named `func(*ecs.World)` entries. Three schedules ship — `UISchedule`, `GameSchedule`, `EngineSchedule` — and `TickFrame` runs them in that order so engine systems see the latest UI and game state for the frame.

## Asset layer

Mesh, texture, material, and glTF loading sit on typed caches stored as resources on the engine world.

`MeshAssets` registers vertex buffers and returns `MeshHandle` indices. Every registered mesh carries its local-space AABB so the bounding-volume overlay and any spatial query can read bounds without re-walking vertices. Built-in primitives (cube, sphere, plane) are registered by `RegisterPrimitives` and exposed as a `Primitives` resource.

`TextureCache` registers GPU textures with full mip chains. Callers pick `TextureSRGB` or `TextureLinear` at register time so base color and emissive maps land in sRGB while normal, metallic-roughness, and occlusion maps stay linear. A 1×1 white texture is registered by default and bound wherever a real texture has not been supplied.

`MaterialRegistry` and `MaterialTextureArrays` keep PBR materials and their texture arrays in a global, indexable form. `RenderMesh` itself is just a `MeshHandle`; the matching `Material` component on the same entity carries the BaseColor, factors, and packed layer indices into the shared sRGB / linear texture arrays. The mesh pass uploads a per-instance `material_id` buffer alongside the per-instance transforms, and the WGSL mesh shader indexes the global material storage buffer through it.

`LoadGltfReader` parses a `.glb` or `.gltf` document, uploads every primitive into the mesh cache, decodes embedded PNG / JPEG images into the texture cache (with auto-classification by sRGB vs linear based on which material slot references each texture), and returns a `LoadedScene` that mirrors the glTF node hierarchy. `SpawnLoadedScene` materializes the scene as ECS entities with `transform.Parent` links so the engine's transform propagation handles world matrices automatically. The load step is pure CPU work; the spawn step writes ECS entities — the split means an app can inspect or transform the loaded data before any entity exists.

IBL is precomputed once per environment: a procedural sky pass writes a cubemap, `pass.IBL` filters it for diffuse and specular reflectance, and the result lands on the mesh pass as bind groups alongside the BRDF LUT.

## Rendering layer

The rendering layer turns ECS state into pixels.

The render graph is a declarative pass DAG. A pass is a struct of function values plus a slot manifest:

```go
type Pass struct {
    Name   string
    Reads  []string
    Writes []string

    State any

    Prepare              func(state any, ctx *PassContext) error
    Execute              func(state any, ctx *PassContext) error
    InvalidateBindGroups func(state any)
    Release              func(state any)
}
```

Apps build passes (`render.AddSkyPass`, `render.AddMeshPass`, `render.AddPostprocessPass`, ...) and bind their `Reads` / `Writes` slot names to typed resource IDs. `Compile` topologically sorts the passes by their dependencies, allocates any transient textures whose handles are still empty, decides clear-vs-load by who writes first, and freezes an execution order. `RenderFrame` then walks the sorted list every frame and invokes each pass's `Prepare` then `Execute`.

The slot indirection means a pass's source code references "color" and "depth" without baking in which concrete textures they map to. A new app can construct its own slot wiring without touching the pass implementation. Adding a pass is one function call plus a slot manifest.

Bind groups are version-tracked. Resources stamp a counter every time their underlying handle changes (transient reallocations on resize, external view replacement each frame). A pass records the version it cached its bind groups against; on a mismatch the pass's `InvalidateBindGroups` runs and rebuilds them. Write-only attachments don't trigger invalidation, so a pass that writes to the swapchain doesn't rebuild on every present.

The standard pipeline writes an HDR scene buffer through the sky, mesh, grid, debug-lines, gizmo, and outline passes, then runs FXAA and an ACES tonemap into the swapchain through the postprocess pass. Lighting is clustered; the cluster bounds and assignment passes run before the mesh pass.

## Feature layer

Built on rendering and the ECS, the feature layer covers everything that is not the base PBR pipeline.

The **retained UI** is a third ECS world. Nodes carry `Color`, `Text`, `Parent`, `Rect`, `Hit`, `Focus`, and text-input buffers as components. The layout system and hit-test system walk this world's tables; the renderer's UI quad and UI text passes iterate the same components from a graph pass. The bridge resource `ui.WorldRef` lets the engine world's render passes see into the UI world.

**Picking** is a render pass that draws entity IDs into an offscreen render target, then reads back the texel under the cursor. The readback completes one or two frames later, asynchronously; the platform layer polls `ProcessPickingReadback` each frame and dispatches a pick result when one is ready.

**Gizmos**, **outlines**, and **debug lines** are separate passes that overlay the scene with translate/rotate/scale handles, selected-entity outlines, and arbitrary debug geometry (normals, bounding volumes, lines registered through the `Lines` resource).

**Animation** has two halves. The glTF loader captures animation clips and channels into `LoadedScene.Animations`. An `AnimationPlayer` component plus a sampling system reads those clips at runtime and writes interpolated transforms into the affected entities' `LocalTransform`. The skinning side of the pipeline is in progress — meshes that ship with skin bindings are loaded but not yet sampled to bone matrices.

## App layer

The `app` package wires the engine world, game world, and UI world together with their cross-world bridges, and offers a small set of lifecycle hooks for the platform layer to invoke.

`app.NewWorlds(renderer)` builds all three worlds and installs `EngineRef` on the game world and `ui.WorldRef` on the engine world. `app.App` is a struct of optional function values — `Initialize`, `ConfigureRenderGraph`, `RunSystems`, `PreRender` — that the platform layer calls at fixed points. Sync helpers (`SyncEngineTransform`, `SyncEngineTranslation`, `SyncEngineRotation`, `DespawnLinked`) move state from game-side entities to engine-side entities through a typed `EngineEntity` link, replacing the scattered `EngineRef.World.Mutate(...)` calls a hand-built app would otherwise sprinkle through its systems.

## Application layer

At the top sit the concrete consumers. `cmd/breakout` is a 2D-ish game with a paddle, ball, bricks, and a HUD; it demonstrates the game-world → engine-world sync pattern and a custom render graph. `cmd/editor` is a 3D scene viewer with pan/orbit camera, glTF loading, picking, outlines, gizmos, and a docked HUD with hierarchy and inspector panels.

Both apps share the same engine binary surface. The web build is the same Go program compiled with `GOOS=js GOARCH=wasm`; the platform layer's `main_js.go` swaps the GLFW event loop for a `requestAnimationFrame` loop driven by JavaScript shims, but the schedules, the render graph, and the asset pipeline are identical.

## What the design refuses

A handful of tradeoffs are deliberate:

- The engine has no scripting layer. Systems are plain Go; the app brings its own.
- Transient textures are not aliased between non-overlapping passes today. Each transient is its own GPU texture. A pooling layer would save memory on large graphs but isn't load-bearing at current graph sizes.
- The renderer is single-threaded. Encoder construction, bind group rebuilds, and per-pass `Prepare` all happen on the main goroutine.
- Skinning is not yet wired through to the mesh pass. `LoadedScene.Animations` is captured and an `AnimationPlayer` component exists, but skeletal vertices are not yet sampled to bone matrices at draw time.

These are scope decisions, not regressions. The remaining chapters cover each layer in depth.
