PYTHON=$(shell which python3)
SOURCES=src/
BIN=bin/
SAMPLES=1000
FEATURES=2

.PHONY: all maze.data compile-examples
all: compile compile-examples maze.data

maze.data:
	@cd data; ${PYTHON} generate_maze.py ${SAMPLES} # Generate $SAMPLES random mazes of depth 1

compile: bin/.dirstamp
	@-futhark-opencl ${SOURCES}prediction.fut -I ${SOURCES} -o ${BIN}predict

compile-examples: compile bin/maze3 bin/maze4

bin/maze3: examples/maze3.fut
	@-futhark-opencl $< -I ${SOURCES} -o $@

bin/maze4: examples/maze4.fut
	@-futhark-opencl $< -I ${SOURCES} -o $@

bin/.dirstamp:
	@mkdir -p bin
	@touch $@

clean:
	@rm -rf bin *.res *.time
