NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out
BIN := $(OUT_DIR)/matmul

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

# -------- Added for Debugging -----------
ifeq ($(DEBUG),1)
    NVCC_FLAGS += -g -G -Xcompiler=-g   # host+device debug
endif
# -----------------------------

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

# convenience alias so `make matmul` still works
.PHONY: matmul clean
.DEFAULT_GOAL := matmul 

matmul: $(BIN)

$(OUT_DIR):
	mkdir -p $@

# build the real binary
$(BIN): matmul.cu | $(OUT_DIR)
	$(NVCC_BASE) $^ -o $@

# convenience alias so `make matmul` still works
.PHONY: matmul
matmul: $(BIN)

#sum: sum.cu 
#	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

#sumprofile: sum
#	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^


#matmul: matmul.cu 
#	mkdir -p $(OUT_DIR)
#	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

#matmulprofile: matmul
#	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

clean:
	rm $(OUT_DIR)/*
