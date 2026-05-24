set windows-shell := ["powershell.exe"]

# Displays the list of available commands
@just:
    just --list

# Runs the named app (default: editor). Example: `just run breakout`.
run $project="editor":
    go run ./cmd/{{project}}

# Builds the named app's desktop binary.
build $project="editor":
    go build ./cmd/{{project}}

# Builds the editor wasm bundle into site/ (Windows).
[windows]
build-wasm:
    $env:GOOS = "js"; $env:GOARCH = "wasm"; go build -o site/main.wasm ./cmd/editor
    Copy-Item "$((go env GOROOT))/lib/wasm/wasm_exec.js" site/wasm_exec.js

# Builds the editor wasm bundle into site/ (Unix).
[unix]
build-wasm:
    GOOS=js GOARCH=wasm go build -o site/main.wasm ./cmd/editor
    cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" site/wasm_exec.js

# Serves site/ on http://localhost:8080
serve:
    go run ./cmd/serve

# Builds the editor wasm bundle and serves site/.
run-wasm: build-wasm
    just serve

# Runs go vet and fails on unformatted files (Windows)
[windows]
check:
    go vet ./...
    $unformatted = (gofmt -l . | Out-String).Trim(); if ($unformatted) { Write-Host $unformatted; exit 1 }

# Runs go vet and fails on unformatted files (Unix)
[unix]
check:
    go vet ./...
    unformatted="$(gofmt -l .)"; if [ -n "$unformatted" ]; then echo "$unformatted"; exit 1; fi

# Formats all Go files
format:
    gofmt -w .

# Runs all tests
test:
    go test ./...

# Runs check + test (use this before pushing)
ci: check test

# Lists all module dependencies with available updates
outdated:
    go list -m -u all

# Shows what `go mod tidy` would change without applying it
tidy-check:
    go mod tidy -diff

# Tidies go.mod / go.sum
tidy:
    go mod tidy

# Runs every read-only check: vet+fmt, tidy diff, outdated, tests
audit: check tidy-check outdated test

# Renders package docs
doc:
    go doc -all ./ecs
    go doc -all ./render
    go doc -all ./app

# Builds the mdbook
book-build:
    just -d docs/book --justfile docs/book/justfile build

# Serves the mdbook locally and opens the browser
book-serve:
    just -d docs/book --justfile docs/book/justfile serve

# Watches the mdbook sources and rebuilds on change
book-watch:
    just -d docs/book --justfile docs/book/justfile watch

# Cleans the mdbook build output
book-clean:
    just -d docs/book --justfile docs/book/justfile clean

# Removes any built binaries (Windows)
[windows]
clean:
    Remove-Item -Force -ErrorAction SilentlyContinue indigo.exe
    Remove-Item -Force -ErrorAction SilentlyContinue editor.exe

# Removes any built binaries (Unix)
[unix]
clean:
    rm -f indigo indigo.exe editor editor.exe

# Displays Go tool version
@versions:
    go version
