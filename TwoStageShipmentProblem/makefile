JULIA = julia
MAIN = main.jl
SIZE ?= 100
VERBOSE ?= false

.PHONY: run clean

run:
	$(JULIA) --project=$(CURDIR) $(MAIN) --size=$(SIZE) --verbose=$(VERBOSE)

clean:
	rm -rf *.log *.tmp