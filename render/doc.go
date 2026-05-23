// Package render is the wgpu-backed renderer. It owns a data-oriented render
// graph whose passes declare the resources they read and write; the graph
// topologically orders them, allocates transient textures, and records GPU
// commands. The package also defines the cameras, lights, materials, and
// resource table the passes operate on. Pass implementations live in
// internal/render/pass.
package render
