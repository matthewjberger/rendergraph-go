package render

import "indigo/transform"

// GizmoMode selects which manipulator overlay to draw around the
// selected entity. None hides the gizmo entirely.
type GizmoMode uint8

const (
	GizmoNone GizmoMode = iota
	GizmoTranslate
	GizmoRotate
	GizmoScale
)

// Gizmos is the engine-world resource that tracks the active gizmo
// mode, the current hovered or dragged axis, and the world-space
// anchors captured at drag-start. The gizmo pass + the gizmo system
// (both in the pass subpackage) read and write this resource.
// Pan-orbit camera reads only Dragging + HoverAxis so its input is
// suppressed while a handle is engaged.
type Gizmos struct {
	Mode GizmoMode

	HoverAxis int

	Dragging bool
	DragAxis uint8
	DragMode GizmoMode

	StartLocal          transform.LocalTransform
	StartGlobalTrans    [3]float32
	AxisWorldDirection  [3]float32
	AxisWorldLengthDrag float32
	InitialT            float32
	StartWorldVector    [3]float32

	PrevLeftDown bool
}

// NewGizmos returns the default Gizmos resource: translate mode,
// nothing hovered, nothing dragging.
func NewGizmos() *Gizmos {
	return &Gizmos{Mode: GizmoTranslate, HoverAxis: -1}
}
