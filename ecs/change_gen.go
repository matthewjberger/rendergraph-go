package ecs

import "unsafe"

func IterChanged1[A any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex])
			}
		}
	}
}

func IterChanged2[A, B any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		bColumn := table.columns[bInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		bSlice := unsafe.Slice((*B)(bColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since || bColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex])
			}
		}
	}
}

func IterChanged3[A, B, C any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		bColumn := table.columns[bInfo.bitIndex]
		cColumn := table.columns[cInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		bSlice := unsafe.Slice((*B)(bColumn.dataPtr), count)
		cSlice := unsafe.Slice((*C)(cColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since || bColumn.changed[arrayIndex] > since || cColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex])
			}
		}
	}
}

func IterChanged4[A, B, C, D any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		bColumn := table.columns[bInfo.bitIndex]
		cColumn := table.columns[cInfo.bitIndex]
		dColumn := table.columns[dInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		bSlice := unsafe.Slice((*B)(bColumn.dataPtr), count)
		cSlice := unsafe.Slice((*C)(cColumn.dataPtr), count)
		dSlice := unsafe.Slice((*D)(dColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since || bColumn.changed[arrayIndex] > since || cColumn.changed[arrayIndex] > since || dColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex])
			}
		}
	}
}

func IterChanged5[A, B, C, D, E any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D, e *E)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	eInfo := mustComponentInfo[E](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		bColumn := table.columns[bInfo.bitIndex]
		cColumn := table.columns[cInfo.bitIndex]
		dColumn := table.columns[dInfo.bitIndex]
		eColumn := table.columns[eInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		bSlice := unsafe.Slice((*B)(bColumn.dataPtr), count)
		cSlice := unsafe.Slice((*C)(cColumn.dataPtr), count)
		dSlice := unsafe.Slice((*D)(dColumn.dataPtr), count)
		eSlice := unsafe.Slice((*E)(eColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since || bColumn.changed[arrayIndex] > since || cColumn.changed[arrayIndex] > since || dColumn.changed[arrayIndex] > since || eColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex], &eSlice[arrayIndex])
			}
		}
	}
}

func IterChanged6[A, B, C, D, E, F any](world *World, extraInclude, exclude Mask, callback func(entity Entity, a *A, b *B, c *C, d *D, e *E, f *F)) {
	world.enterIter()
	defer world.leaveIter()
	aInfo := mustComponentInfo[A](world)
	bInfo := mustComponentInfo[B](world)
	cInfo := mustComponentInfo[C](world)
	dInfo := mustComponentInfo[D](world)
	eInfo := mustComponentInfo[E](world)
	fInfo := mustComponentInfo[F](world)
	since := world.lastTick
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
		aColumn := table.columns[aInfo.bitIndex]
		bColumn := table.columns[bInfo.bitIndex]
		cColumn := table.columns[cInfo.bitIndex]
		dColumn := table.columns[dInfo.bitIndex]
		eColumn := table.columns[eInfo.bitIndex]
		fColumn := table.columns[fInfo.bitIndex]
		aSlice := unsafe.Slice((*A)(aColumn.dataPtr), count)
		bSlice := unsafe.Slice((*B)(bColumn.dataPtr), count)
		cSlice := unsafe.Slice((*C)(cColumn.dataPtr), count)
		dSlice := unsafe.Slice((*D)(dColumn.dataPtr), count)
		eSlice := unsafe.Slice((*E)(eColumn.dataPtr), count)
		fSlice := unsafe.Slice((*F)(fColumn.dataPtr), count)
		for arrayIndex := 0; arrayIndex < count; arrayIndex++ {
			if aColumn.changed[arrayIndex] > since || bColumn.changed[arrayIndex] > since || cColumn.changed[arrayIndex] > since || dColumn.changed[arrayIndex] > since || eColumn.changed[arrayIndex] > since || fColumn.changed[arrayIndex] > since {
				callback(table.Entities[arrayIndex], &aSlice[arrayIndex], &bSlice[arrayIndex], &cSlice[arrayIndex], &dSlice[arrayIndex], &eSlice[arrayIndex], &fSlice[arrayIndex])
			}
		}
	}
}
