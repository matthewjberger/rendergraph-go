package render

// CameraMarker tags an entity as a scene camera that should render
// a wireframe gizmo from its transform. Distinct from the active
// editor camera (which lives as a resource) so multiple scene
// cameras can coexist and be selected individually.
type CameraMarker struct{}
