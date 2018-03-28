PYTHON=$(shell which python3)
SOURCES=src/
BIN=bin/
SAMPLES=1000
FEATURES=2

.PHONY: all
all: compile

maze: maze.res
	cat $<

maze.res: maze.data compile
	echo "Maze prediction percentage:" > $@
	time cat data/maze_x_n${SAMPLES}_features${FEATURES}.txt data/maze_y_n${SAMPLES}_features${FEATURES}.txt | bin/predict -t maze.time >> $@
	echo "Execution time (excl. data loading):" >> $@
	cat maze.time >> $@

maze.data:
	@cd data; ${PYTHON} generate_maze.py ${SAMPLES} # Generate $SAMPLES random mazes of depth 1

compile: bin/.dirstamp
	@-futhark-opencl ${SOURCES}prediction.fut -I ${SOURCES} -o ${BIN}predict

bin/.dirstamp:
	@mkdir -p bin
	@touch $@

clean:
	@rm -rf bin *.res *.time
