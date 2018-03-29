# OpenCL backend

A backend for executing volr models. Running on OpenCL and programmed in Futhark

Heavily inspired by [Martin Elsmans port](https://github.com/melsman/neural-networks-and-deep-learning) of the [Networks and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html) article by Michael Nielsen.

## Setup

### Requirements

* Python > 3.x: https://www.python.org/
* Futhark >= 0.4.0: https://github.com/diku-dk/futhark
* OpenCL (used by Futhark): https://www.khronos.org/opencl/

To execute the examples:
* Stack for Haskell: https://docs.haskellstack.org/en/stable/README/
* Maze generator: https://github.com/volr/maze

### Installation

Simply enter the project and execute `make`. This will compile the library,
the two examples and the data for the maze examples.

## Maze example

In the `data` folder two maze data files will be generated
(`maze_x_n1000_features2.txt` and `maze_y_n1000_features2.txt`). `cat`ting
these into the respective binary will feed the data into the network.

    cat data/*x_n1000* data/*y_n1000* | bin/maze4

For benchmarking, Futhark gives the possibility to output the execution time
(excluding I/O) with the `-t` flag (see [the Futhark manual](https://futhark.readthedocs.io/en/latest/usage.html)):

    cat data/*x_n1000* data/*y_n1000* | bin/maze4 -t /dev/stdout

Note that the time is given in microseconds (10<sup>-6</sup>s).

Finally, Futhark also allows to execute a number of runs with the `-r` flag to
average on multiple runs and avoid warmup time deviations:

    cat data/*x_n1000* data/*y_n1000* | bin/maze4 -r 10 -t /dev/stdout

## Credits
A lot of this code was written by [Martin Elsmans port](https://github.com/melsman/)
who, in turn, got the idea from the [Networks and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html) article by Michael Nielsen.

Contact: jensegholm@protonmail.com
