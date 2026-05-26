package main

import (
	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
	"github.com/matthewjberger/indigo/transform"
	"github.com/matthewjberger/indigo/ui"
)

type pickCycleState struct {
	LastLeaf   ecs.Entity
	LastRoot   ecs.Entity
	CycleDepth int
	AltHeld    bool
	ShiftHeld  bool
}

func installPickCycleState(engine *ecs.World) {
	ecs.SetResource(engine, pickCycleState{})
}

func recordPickModifiers(engine *ecs.World, altHeld, shiftHeld bool) {
	state := ecs.MustResource[pickCycleState](engine)
	state.AltHeld = altHeld
	state.ShiftHeld = shiftHeld
}

func resetPickCycle(engine *ecs.World) {
	state := ecs.MustResource[pickCycleState](engine)
	state.LastLeaf = ecs.Entity{}
	state.LastRoot = ecs.Entity{}
	state.CycleDepth = 0
}

func handlePickResult(worlds app.Worlds, pickedID uint32) {
	gizmo := *ecs.MustResource[*render.Gizmos](worlds.Engine)
	if gizmo != nil && (gizmo.Dragging || gizmo.DraggingPlane) {
		return
	}
	if worlds.UI != nil {
		pointer := ecs.MustResource[ui.PointerState](worlds.UI)
		if pointer.OverUI {
			return
		}
	}
	if gizmo != nil && (gizmo.HoverAxis >= 0 || gizmo.HoverPlane >= 0) {
		return
	}
	applySelection(worlds.Engine, pickedID)
}

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
	if proxy, ok := ecs.Get[render.PickProxy](engine, leaf); ok && proxy != nil {
		leaf = proxy.Target
	}

	target := resolveCycleTarget(engine, state, leaf)
	if state.ShiftHeld {
		toggleSelection(engine, target)
	} else {
		applyEntitySelection(engine, target)
	}
}

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

func findEntityByID(engine *ecs.World, id uint32) (ecs.Entity, bool) {
	var picked ecs.Entity
	found := false
	scan := func(mask ecs.Mask) {
		if found {
			return
		}
		engine.ForEach(mask, 0, func(e ecs.Entity, _ *ecs.Archetype, _ int) {
			if !found && e.ID == id {
				picked = e
				found = true
			}
		})
	}
	scan(ecs.MustMaskOf[asset.RenderMesh](engine))
	scan(ecs.MustMaskOf[asset.SkinnedMesh](engine))
	return picked, found
}

func applyEntitySelection(engine *ecs.World, entity ecs.Entity) {
	clearSelection(engine)
	ecs.Add[render.Selected](engine, entity)
}

func toggleSelection(engine *ecs.World, entity ecs.Entity) {
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
