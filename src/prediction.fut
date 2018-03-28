
---------------------------------------
-- Stochastic gradient descent learning
-- Author: Jens Egholm <jensegholm@protonmail.com>
-- Heavily inspired by Martin Elsmans implementation:
--    https://github.com/melsman/neural-networks-and-deep-learning
-- License: GPLv3
---------------------------------------

import "/futlib/linalg"
import "/futlib/math"
import "/futlib/radix_sort"
import "neuralnetwork"

module random = import "/futlib/random"
module rng_engine = random.minstd_rand0
module array  = import "/futlib/array"

module N = NeuralNetwork(f64)

module N3 = N.Network3({
  let size1 = 2
  let size2 = 10
  let output = 2
})

let main [n] [m] (x: [n][m]N.t) (y: [n]i32) : N.t =
  let split = i32.f64(f64.floor(f64.i32(n) * 0.8))
  let training_x = x[0:split]
  let training_y = y[0:split]
  let test_x = x[split:n]
  let test_y = y[split:n]
  in N3.run (training_x, training_y, test_x, test_y)
