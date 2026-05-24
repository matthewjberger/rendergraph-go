package ecs

import "unsafe"

func Iter1[A any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	include := extraInclude | aInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex])
		}
	}
}

func Iter2[A, B any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	include := extraInclude | aInfo.mask | bInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		bSlice := unsafe.Slice((*B)(table.columns[bInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex])
		}
	}
}

func Iter3[A, B, C any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	include := extraInclude | aInfo.mask | bInfo.mask | cInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		bSlice := unsafe.Slice((*B)(table.columns[bInfo.bitIndex].dataPtr), count)
		cSlice := unsafe.Slice((*C)(table.columns[cInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex])
		}
	}
}

func Iter4[A, B, C, D any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	include := extraInclude | aInfo.mask | bInfo.mask | cInfo.mask | dInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		bSlice := unsafe.Slice((*B)(table.columns[bInfo.bitIndex].dataPtr), count)
		cSlice := unsafe.Slice((*C)(table.columns[cInfo.bitIndex].dataPtr), count)
		dSlice := unsafe.Slice((*D)(table.columns[dInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex])
		}
	}
}

func Iter5[A, B, C, D, E any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D, e *E)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	eInfo := mustComponentInfo[E](world)
	include := extraInclude | aInfo.mask | bInfo.mask | cInfo.mask | dInfo.mask | eInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		bSlice := unsafe.Slice((*B)(table.columns[bInfo.bitIndex].dataPtr), count)
		cSlice := unsafe.Slice((*C)(table.columns[cInfo.bitIndex].dataPtr), count)
		dSlice := unsafe.Slice((*D)(table.columns[dInfo.bitIndex].dataPtr), count)
		eSlice := unsafe.Slice((*E)(table.columns[eInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex], &eSlice[arrayIndex])
		}
	}
}

func Iter6[A, B, C, D, E, F any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D, e *E, f *F)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	eInfo := mustComponentInfo[E](world)
	fInfo := mustComponentInfo[F](world)
	include := extraInclude | aInfo.mask | bInfo.mask | cInfo.mask | dInfo.mask | eInfo.mask | fInfo.mask
	for _, tableIndex := range world.cachedTables(include) {
		table := world.tables[tableIndex]
		if table.Mask&exclude != 0 {
			continue
		}
		count := len(table.Entities)
		if count == 0 {
			continue
		}
		aSlice := unsafe.Slice((*A)(table.columns[aInfo.bitIndex].dataPtr), count)
		bSlice := unsafe.Slice((*B)(table.columns[bInfo.bitIndex].dataPtr), count)
		cSlice := unsafe.Slice((*C)(table.columns[cInfo.bitIndex].dataPtr), count)
		dSlice := unsafe.Slice((*D)(table.columns[dInfo.bitIndex].dataPtr), count)
		eSlice := unsafe.Slice((*E)(table.columns[eInfo.bitIndex].dataPtr), count)
		fSlice := unsafe.Slice((*F)(table.columns[fInfo.bitIndex].dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex], &eSlice[arrayIndex], &fSlice[arrayIndex])
		}
	}
}
