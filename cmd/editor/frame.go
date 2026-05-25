package main

import (
	"github.com/matthewjberger/indigo/app"
	"github.com/matthewjberger/indigo/ecs"
	"github.com/matthewjberger/indigo/internal/render/pass"
	"github.com/matthewjberger/indigo/render"
	"github.com/matthewjberger/indigo/render/asset"
)

func updateEditorFrame(worlds app.Worlds, renderer *render.Renderer, demo *app.App, delta float32) bool {
	drainKhronosPending(worlds, renderer)
	ecs.MustResource[asset.LoadingQueueResource](worlds.Engine).Queue.Drain(renderer.Queue)
	syncUiPointer(worlds)
	ctx := newHudContext(worlds)
	ctx.refreshHudLayout()
	ctx.updateTreeScroll()

	app.TickFrame(worlds, demo, delta)
	if ctx.Hud.FramePending {
		frameCameraOnRoots(worlds.Engine, ctx.Hud.ModelRoots)
		ctx.Hud.FramePending = false
	}
	handleRightClick(worlds)
	driveTextInputs(worlds)
	handleUiClicks(worlds)
	ctx.refreshFps()
	ctx.refreshLoadingProgress()
	ctx.refreshModeButtons()
	ctx.refreshViewerCheckbox()
	ctx.refreshMenuPopups()
	ctx.refreshInteractiveHovers()
	ctx.refreshEntityTree()
	ctx.refreshInspector()
	ctx.updateInspectorCaret()
	ctx.refreshKhronosBrowser()

	return ctx.Hud.RequestExit
}

func finishEditorFrame(worlds app.Worlds, renderer *render.Renderer) {
	pass.ProcessPickingReadback(renderer, worlds.Engine)
	if picking := ecs.MustResource[*pass.Picking](worlds.Engine); (*picking).Result != nil {
		result := (*picking).Result
		(*picking).Result = nil
		handlePickResult(worlds, result.EntityID)
	}
	app.PostFrame(worlds)
}
