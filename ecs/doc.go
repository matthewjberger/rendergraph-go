//go:generate go run ../internal/gen_iter

// Package ecs is a data-oriented entity component system. Entities are
// generational handles, components are plain structs held in per-archetype
// columns, and a query iterates the columns of every archetype whose
// component set satisfies it. Hot-path iteration via the generated Iter
// helpers materializes typed column views once per archetype.
package ecs
