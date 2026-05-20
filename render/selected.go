package render

// Selected is a tag component (zero-sized) that marks an engine
// entity as a member of the current editor selection. Entities with
// this component get drawn into the selection mask each frame and
// receive an outline overlay from the outline post-process pass.
//
// Selection is engine-scope state with no policy attached: apps
// decide who adds and removes the tag. The editor wires up the
// picking result so left-clicking an entity becomes the new
// selection.
type Selected struct{}
