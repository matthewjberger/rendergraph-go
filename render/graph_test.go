package render

import (
	"testing"

	"github.com/matthewjberger/indigo/ecs"
)

func externalColor(name string) ResourceDescriptor {
	return ResourceDescriptor{Name: name, Kind: ResourceKindExternalColor}
}

func recordingPass(name string, reads, writes []string, log *[]string) *Pass {
	return &Pass{
		Name:   name,
		Reads:  reads,
		Writes: writes,
		Prepare: func(_ *PassContext) error {
			*log = append(*log, name)
			return nil
		},
	}
}

func TestAddPassRejectsUnboundReadSlot(t *testing.T) {
	graph := NewGraph()
	id := graph.AddColorTexture(externalColor("color"))

	pass := &Pass{Name: "p", Reads: []string{"input"}}
	err := graph.AddPass(pass, []SlotBinding{{Slot: "wrong", ResourceID: id}})
	if err == nil {
		t.Fatal("expected error for unbound read slot")
	}
}

func TestAddPassRejectsUnboundWriteSlot(t *testing.T) {
	graph := NewGraph()
	id := graph.AddColorTexture(externalColor("color"))

	pass := &Pass{Name: "p", Writes: []string{"output"}}
	err := graph.AddPass(pass, []SlotBinding{{Slot: "other", ResourceID: id}})
	if err == nil {
		t.Fatal("expected error for unbound write slot")
	}
}

func TestResourceByNameReturnsFirstMatch(t *testing.T) {
	graph := NewGraph()
	first := graph.AddColorTexture(externalColor("color"))
	graph.AddColorTexture(externalColor("other"))

	if got := graph.ResourceByName("color"); got != first {
		t.Fatalf("ResourceByName(color) = %v, want %v", got, first)
	}
	if got := graph.ResourceByName("missing"); got != 0 {
		t.Fatalf("ResourceByName(missing) = %v, want 0", got)
	}
}

func TestExecuteRespectsWriteAfterWriteOrdering(t *testing.T) {
	graph := NewGraph()
	color := graph.AddColorTexture(externalColor("color"))

	var log []string
	first := recordingPass("first", nil, []string{"out"}, &log)
	second := recordingPass("second", nil, []string{"out"}, &log)
	reader := recordingPass("reader", []string{"in"}, nil, &log)

	if err := graph.AddPass(first, []SlotBinding{{Slot: "out", ResourceID: color}}); err != nil {
		t.Fatal(err)
	}
	if err := graph.AddPass(second, []SlotBinding{{Slot: "out", ResourceID: color}}); err != nil {
		t.Fatal(err)
	}
	if err := graph.AddPass(reader, []SlotBinding{{Slot: "in", ResourceID: color}}); err != nil {
		t.Fatal(err)
	}

	if err := graph.Compile(nil); err != nil {
		t.Fatalf("Compile: %v", err)
	}
	world := ecs.New()
	if err := graph.Execute(nil, nil, world, nil); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	want := []string{"first", "second", "reader"}
	for index := range want {
		if log[index] != want[index] {
			t.Fatalf("log = %v, want %v", log, want)
		}
	}
}

func TestExecuteBeforeCompileErrors(t *testing.T) {
	graph := NewGraph()
	id := graph.AddColorTexture(externalColor("color"))
	pass := recordingPass("p", nil, []string{"out"}, &[]string{})
	if err := graph.AddPass(pass, []SlotBinding{{Slot: "out", ResourceID: id}}); err != nil {
		t.Fatal(err)
	}

	world := ecs.New()
	if err := graph.Execute(nil, nil, world, nil); err == nil {
		t.Fatal("Execute on uncompiled graph should error")
	}
}

func TestSetExternalTextureBumpsVersion(t *testing.T) {
	var resources Resources
	id := resources.Register(externalColor("swap"))
	if resources.Version(id) != 0 {
		t.Fatalf("initial version = %d, want 0", resources.Version(id))
	}
	resources.SetExternalTexture(id, nil, 1280, 720)
	if resources.Version(id) != 1 {
		t.Fatalf("after set version = %d, want 1", resources.Version(id))
	}
	resources.SetExternalTexture(id, nil, 1920, 1080)
	if resources.Version(id) != 2 {
		t.Fatalf("after second set version = %d, want 2", resources.Version(id))
	}
}

func TestRegisterAssignsSequentialIDs(t *testing.T) {
	var resources Resources
	first := resources.Register(externalColor("a"))
	second := resources.Register(externalColor("b"))
	third := resources.Register(externalColor("c"))
	if first != 0 || second != 1 || third != 2 {
		t.Fatalf("ids = %d %d %d, want 0 1 2", first, second, third)
	}
}

func TestInvalidateBindGroupsCalledOnVersionChange(t *testing.T) {
	graph := NewGraph()
	id := graph.AddColorTexture(externalColor("color"))

	invalidations := 0
	pass := &Pass{
		Name:                 "reader",
		Reads:                []string{"in"},
		InvalidateBindGroups: func() { invalidations++ },
	}
	if err := graph.AddPass(pass, []SlotBinding{{Slot: "in", ResourceID: id}}); err != nil {
		t.Fatal(err)
	}
	if err := graph.Compile(nil); err != nil {
		t.Fatal(err)
	}

	world := ecs.New()
	graph.Resources.SetExternalTexture(id, nil, 1, 1)
	if err := graph.Execute(nil, nil, world, nil); err != nil {
		t.Fatal(err)
	}
	if invalidations != 1 {
		t.Fatalf("invalidations after first version change = %d, want 1", invalidations)
	}

	if err := graph.Execute(nil, nil, world, nil); err != nil {
		t.Fatal(err)
	}
	if invalidations != 1 {
		t.Fatalf("invalidations should not fire when version is stable, got %d", invalidations)
	}

	graph.Resources.SetExternalTexture(id, nil, 2, 2)
	if err := graph.Execute(nil, nil, world, nil); err != nil {
		t.Fatal(err)
	}
	if invalidations != 2 {
		t.Fatalf("invalidations after second version change = %d, want 2", invalidations)
	}
}
