NVCC    ?= nvcc
ARCH    ?= native
# -O3 affects host code only; device code is ptxas -O3 by default.
# -lineinfo enables ncu source correlation at no runtime cost.
NVFLAGS ?= -O3 -lineinfo -arch=$(ARCH) -I.
PYTHON  ?= python3

SRC_DIR   := src
BUILD_DIR := build
RESULTS   := results
DOCS_DIR  := docs

SRCS := $(sort $(wildcard $(SRC_DIR)/*.cu))
BINS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SRCS))

.PHONY: all build run profile metrics-list clean

all: build

build: $(BINS)

# The 'build' target shares a name with the output directory, so the directory
# is created in the recipe rather than as a prerequisite.
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu common.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) $< -o $@

# Combined summary table; exits non-zero if any before/after pair disagrees.
run: $(BINS)
	@$(firstword $(BINS)) header
	@fail=0; for b in $(BINS); do $$b row || fail=1; done; exit $$fail

# Full ncu sweep; writes reports, analysis and prompts to $(RESULTS)/
profile: $(BINS)
	$(PYTHON) run_ncu.py

# Device metric list backing docs/ncu-metrics.md; regenerate after a driver or
# GPU change, since unsupported counters are silently dropped by ncu.
metrics-list:
	@mkdir -p $(DOCS_DIR)
	ncu --query-metrics > $(DOCS_DIR)/ncu-metrics-device.txt
	@echo "wrote $(DOCS_DIR)/ncu-metrics-device.txt ($$(wc -l < $(DOCS_DIR)/ncu-metrics-device.txt) lines)"

clean:
	rm -rf $(BUILD_DIR) $(RESULTS)
