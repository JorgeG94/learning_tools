CC = nvc
CFLAGS = -mp=gpu,multicore -O3
TARGET = maps
SRC = maps.c

NELEM ?= 4096

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run-cpu: $(TARGET)
	OMP_TARGET_OFFLOAD=disabled ./$(TARGET) $(NELEM)

run-gpu: $(TARGET)
	OMP_TARGET_OFFLOAD=mandatory ./$(TARGET) $(NELEM)

clean:
	rm -f $(TARGET)

.PHONY: all run-cpu run-gpu clean
