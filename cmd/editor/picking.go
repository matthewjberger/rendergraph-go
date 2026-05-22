package main

import (
	"indigo/app"
	"indigo/ecs"
	"indigo/render"
	"indigo/render/asset"
	"indigo/transform"
	"indigo/ui"
)

// pickCycleState tracks the click-to-cycle picking sequence so
// repeated clicks on the same leaf walk further down the parent
// chain. last_pick_root + last_pick_leaf identify the "current
// chain"; cycle_depth is the index into that chain that gets
// selected on the next click.
type pickCycleState struct {
	LastLeaf   ecs.Entity
	LastRoot   ecs.Entity
	CycleDepth int
	AltHeld    bool
	ShiftHeld  bool
}

// installPickCycleState seeds the resource on the engine world.
func installPickCycleState(engine *ecs.World) {
	ecs.SetResource(engine, pickCycleState{})
}

// recordPickModifiers stashes alt/shift state at click time. The
// pick result arrives a frame later (GPU readback), so we cache
// the modifiers at request time.
func recordPickModifiers(engine *ecs.World, altHeld, shiftHeld bool) {
	state := ecs.MustResource[pickCycleState](engine)
	state.AltHeld = altHeld
	state.ShiftHeld = shiftHeld
}

// resetPickCycle clears the chain-walk state. Called on undo /
// redo, mode changes, or empty-area clicks where the next click
// should start a fresh cycle.
func resetPickCycle(engine *ecs.World) {
	state := ecs.MustResource[pickCycleState](engine)
	state.LastLeaf = ecs.Entity{}
	state.LastRoot = ecs.Entity{}
	state.CycleDepth = 0
}

// handlePickResult routes the pick result into the cycle resolver,
// except when a gizmo drag is active, the pointer is over UI, or
// the pointer is over a gizmo axis - in those cases the pick lands
// on whatever scene mesh was behind the overlay and changing
// selection would be wrong.
func handlePickResult(worlds app.Worlds, pickedID uint32) {
	gizmo := *ecs.MustResource[*render.Gizmos](worlds.Engine)
	if gizmo != nil && gizmo.Dragging {
		return
	}
	if worlds.UI != nil {
		pointer := ecs.MustResource[ui.PointerState](worlds.UI)
		if pointer.OverUI {
			return
		}
	}
	if gizmo != nil && gizmo.HoverAxis >= 0 {
		return
	}
	applySelection(worlds.Engine, pickedID)
}

// applySelection resolves the pick into a single selected entity
// following the click-to-cycle rule: first click on a new leaf
// selects the leaf's group root; subsequent clicks on the same
// leaf walk one level deeper through the chain each time, wrapping
// back to root after passing the leaf. Alt skips the walk and
// selects the leaf directly. Shift toggles instead of replacing.
func applySelection(engine *ecs.World, pickedID uint32) {
	state := ecs.MustResource[pickCycleState](engine)
	if pickedID == 0 {
		clearSelection(engine)
		resetPickCycle(engine)
		return
	}

	leaf, found := findEntityByID(engine, pickedID)
	if !found {
		clearSelection(engine)
		resetPickCycle(engine)
		return
	}

	target := resolveCycleTarget(engine, state, leaf)
	if state.ShiftHeld {
		toggleSelection(engine, target)
	} else {
		applyEntitySelection(engine, target)
	}
}

// resolveCycleTarget picks the entity from the chain that the
// click should resolve to. Updates pickCycleState in place.
func resolveCycleTarget(engine *ecs.World, state *pickCycleState, leaf ecs.Entity) ecs.Entity {
	if state.AltHeld {
		state.LastLeaf = leaf
		state.LastRoot = leaf
		state.CycleDepth = 0
		return leaf
	}
	root := transform.FindGroupRoot(engine, leaf)
	chain := transform.ChainFromRootToLeaf(engine, root, leaf)
	if len(chain) == 0 {
		state.LastLeaf = leaf
		state.LastRoot = leaf
		state.CycleDepth = 0
		return leaf
	}
	cycling := state.LastLeaf == leaf && state.LastRoot == root
	depth := 0
	if cycling {
		next := state.CycleDepth + 1
		if next >= len(chain) {
			next = 0
		}
		depth = next
	}
	state.LastLeaf = leaf
	state.LastRoot = root
	state.CycleDepth = depth
	return chain[depth]
}

// findEntityByID returns the RenderMesh-bearing entity whose
// runtime ID matches the GPU pick result.
func findEntityByID(engine *ecs.World, id uint32) (ecs.Entity, bool) {
	renderMask := ecs.MustMaskOf[asset.RenderMesh](engine)
	var picked ecs.Entity
	found := false
	engine.ForEach(renderMask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
		if !found && e.ID == id {
			picked = e
			found = true
		}
	})
	return picked, found
}

// applyEntitySelection clears every existing Selected tag and tags
// the supplied entity directly. Used by the entity tree to select
// any named entity (including the Sun light, which has no
// RenderMesh and would be missed by ID-only [applySelection]).
func applyEntitySelection(engine *ecs.World, entity ecs.Entity) {
	clearSelection(engine)
	if entity.ID == 0 {
		return
	}
	ecs.Add[render.Selected](engine, entity)
}

// toggleSelection flips the Selected tag on entity without
// touching other selections (multi-select via shift-click).
func toggleSelection(engine *ecs.World, entity ecs.Entity) {
	if entity.ID == 0 {
		return
	}
	if ecs.Has[render.Selected](engine, entity) {
		ecs.Remove[render.Selected](engine, entity)
		return
	}
	ecs.Add[render.Selected](engine, entity)
}

func clearSelection(engine *ecs.World) {
	selectedMask := ecs.MustMaskOf[render.Selected](engine)
	var toDeselect []ecs.Entity
	engine.ForEach(selectedMask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
		toDeselect = append(toDeselect, e)
	})
	for _, e := range toDeselect {
		ecs.Remove[render.Selected](engine, e)
	}
}
