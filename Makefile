.PHONY: test check verify

# The pool package uses go:linkname to access runtime scheduling
# primitives (dropg, readgstatus) for zero-overhead goroutine parking.
# Go 1.26 restricts these by default; -checklinkname=0 preserves access.
LDFLAGS := -ldflags='-checklinkname=0'

# check runs mechanical enforcement of manifest-first contract.
# See puter/AGENTS.md and puter/GAPS.md §6.5 for the rules.
check:
	@bash "$(CURDIR)/scripts/check_banned.sh"

test:
	go test $(LDFLAGS) -v ./...

# verify is the gate: banned-pattern check first, then tests.
verify: check test