# rendergraph-go

A data-oriented Go game engine: archetype ECS, a `wgpu`-backed render
graph with declarative passes, dual-world simulation/render separation,
free-function systems wired through a named schedule. Runs natively
(GLFW) and on the web (WebAssembly + canvas).

The architecture is ported from
[nightshade](https://github.com/matthewjberger/nightshade); the ECS
layer is [`freecs-go`](https://github.com/matthewjberger/freecs-go),
vendored inline as `ecs/` so the engine and its storage evolve
together. Architecture notes live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What's in the demo

The default scene spawns a 5×5 grid of spinning primitives (cubes,
triangles, quads cycling) over a procedural sky and ground grid,
viewed through a pan-orbit camera, lit by a directional sun with
Lambertian shading, antialiased with FXAA. Each piece is one pass in
the render graph; the topo sort orders them automatically from their
declared reads and writes.

Controls: left-drag to orbit, right-drag to pan, scroll to zoom.
`G` toggles the grid, `S` toggles the sky, `F` toggles FXAA. `Esc`
quits.

## Quickstart

```
just run
```

`just --list` for the rest.

## Provenance

- [`go-template`](https://github.com/matthewjberger/go-template). Module layout, justfile, dual MIT/Apache license.
- [`freecs-go`](https://github.com/matthewjberger/freecs-go). Vendored inline as `ecs/`.
- [`wgpu-example-go`](https://github.com/matthewjberger/wgpu-example-go). GLFW + cogentcore/webgpu surface setup, platform glue.
- [`nightshade`](https://github.com/matthewjberger/nightshade). Engine design being ported.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
