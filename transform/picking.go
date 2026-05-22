package transform

import "indigo/ecs"

// ChainFromRootToLeaf returns the ordered list of entities from
// root (inclusive) down to leaf (inclusive). leaf must be a
// descendant of root or the result is empty. The first element is
// root, the last is leaf, and consecutive pairs are parent/child.
//
// Used by hierarchical picking: a first click on a glTF cluster
// selects chain[0] (the group root), each subsequent click on the
// same leaf advances one step deeper, and depth wraps back to 0
// after passing the leaf.
func ChainFromRootToLeaf(world *ecs.World, root, leaf ecs.Entity) []ecs.Entity {
	if root == leaf {
		return []ecs.Entity{root}
	}
	reverse := []ecs.Entity{leaf}
	current := leaf
	for current != root {
		parent, ok := ecs.Get[Parent](world, current)
		if !ok {
			return nil
		}
		current = parent.Entity
		reverse = append(reverse, current)
		if len(reverse) > MaxHierarchyDepth {
			return nil
		}
	}
	for i, j := 0, len(reverse)-1; i < j; i, j = i+1, j-1 {
		reverse[i], reverse[j] = reverse[j], reverse[i]
	}
	return reverse
}
