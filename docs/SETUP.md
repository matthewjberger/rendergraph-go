# Setup

How to wire indigo into a new app. This page documents the
contract that the editor and breakout apps follow; if you fork
indigo you can copy and trim them.

## Required boilerplate

A native indigo app does five things in `main`:

```go
1. Initialise GLFW + create a window
2. Build the wgpu instance + surface
3. Call render.NewRenderer
4. Call app.NewWorlds — builds the engine + game + UI worlds
5. Run the render loop: TickFrame, renderer.Render, PostFrame
```

See `cmd/editor/main.go` and `cmd/breakout/main.go` for the full
template. The wasm-target equivalent lives in `main_js.go` — same
shape, different windowing entrypoint.

## What NewWorlds bakes in

`app.NewWorlds(renderer)` registers the engine-side component
types (transforms, RenderMesh, SkinnedMesh, Material, Light,
CameraMarker, PickProxy, Selected, Name, AnimationPlayer),
creates the asset registries (MeshAssets, SkinnedMeshAssets,
TextureCache, MaterialTextureArrays, MaterialRegistry), builds
the directional / spot / point shadow systems, builds the IBL
system, allocates the Lines overlay buffer, and sets the standard
resources (Renderer, Input, Graphics, transform state, command
queues).

This is opinionated. If you want a single-world app, no shadows,
or no IBL, fork `app/setup.go` rather than calling NewWorlds.
Issue #32 tracks making it composable.

## Engine schedule

After NewWorlds returns, push systems onto `worlds.EngineSchedule`
in roughly this order:

1. `render.UpdateGraphicsToggles` — keyboard toggles for grid /
   sky / FXAA / bounds / normals / skeletons.
2. `render.UpdatePanOrbitCamera` (or your own controller).
3. `asset.UpdateAnimationPlayers` — advance animation clocks,
   write joint-node LocalTransforms.
4. `transform.UpdateGlobalTransforms` — propagate
   LocalTransform * Parent.Global -> GlobalTransform.
5. `pass.UploadSkinBones` — push the per-frame bone matrices to
   each Skin's GPU buffer (the compute pass that multiplies them
   by inverse-bind matrices runs in the render graph).
6. `pass.UpdateBoundingVolumeLines` / `UpdateNormalLines` /
   `UpdateSkeletonLines` — debug overlay generators; no-ops when
   the matching `Graphics.Show*` flag is false.
7. Your own game systems.

Ordering matters: animations must run before transform
propagation, and propagation before skin upload. If you skip any,
you'll see stale poses one frame behind.

## Render graph

`app.App.ConfigureRenderGraph` is the hook where you register
render passes. The order you call `pass.Add*Pass` is the order
the graph executes (for passes that aren't already dependency-
ordered via slot reads/writes).

The editor's pass order is documented in `cmd/editor/setup.go`:

```
skinning_compute -> sky -> shadow (directional / spot / point)
-> skinned_mesh -> mesh -> picking -> selection_mask -> grid
-> lines -> outline -> ssao -> oit_mesh -> oit_composite
-> auto_exposure -> bloom -> postprocess -> fxaa -> gizmo
-> ui_quad -> ui_text -> present
```

Breakout uses a trimmed subset (no sky / picking / selection /
outline / OIT / grid / lines / gizmo / skinned_mesh). See
`cmd/breakout/setup.go`.

Every pass constructor returns `(*render.Pass, error)`. A nil
error means the pass is registered with the renderer's graph;
you don't need to keep the pointer.

## Compile-time caps

A handful of buffer-shape decisions are still compile-time:
`NumShadowCascades`, `ShadowMapSize`, `MaxLightsBuffer`,
`MaxLightsPerCluster`, `MaxSpotShadows`, `MaxPointShadows`,
`SpotShadowAtlasSize`, `selectionMaskMaxEntities`, `BrdfLutSize`,
`PrefilteredSize`, `IrradianceSamples`, `ProceduralCubemapSize`.

Each lives next to the pass that uses it. Issue #31 tracks pulling
them into a single `EngineConfig` struct so apps can override at
startup.

## Runtime tunables

Everything that is safe to flip per frame lives on the
`render.Graphics` resource (sky / grid / FXAA / bounds / normals
/ skeletons toggles, exposure, bloom intensity / threshold, SSAO
radius / bias / intensity / sample count, cull on/off + min pixel
size, debug-line styling). `render.DefaultGraphics()` returns the
shipped defaults. Mutate the resource from any system —
`UpdateGraphicsToggles` is the canonical example.

## Picking

The renderer writes an `entity_id` texture as a side-effect of the
mesh + OIT passes. `pass.AddPickingPass` adds a 1×1 readback at
the cursor. Clicks resolve via `findEntityByID` in the editor's
`picking.go`; if the hit entity carries a `PickProxy` component
the selection is redirected to `PickProxy.Target` (used for
camera gizmos).

## Loading glTF

`asset.LoadGltfReader` / `LoadGltfFile` parses a glb / glTF into a
`LoadedScene`. `asset.SpawnLoadedScene` instantiates it onto an
engine world. Lights and cameras attached to nodes (via
KHR_lights_punctual or the standard `cameras` array) come back as
`LoadedLight` / `LoadedCamera` entries on the scene; spawn them
with the `attachSceneLights` / `attachSceneCameras` helpers in
`cmd/editor/loader.go`.
