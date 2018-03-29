
---------------------------------------
-- Stochastic gradient descent learning
-- Author: Jens Egholm <jensegholm@protonmail.com>
-- Heavily inspired by Martin Elsmans implementation:
--    https://github.com/melsman/neural-networks-and-deep-learning
-- License: GPLv3
---------------------------------------

import "neuralnetwork"
import "prediction"

module N = Network3 (f32) {
  let size1 = 2
  let size2 = 10
  let output = 2
  let learning_rate = 0.5
}

module P = Predict (N)

let main [n] [m] (x: [n][m]N.t) (y: [n]i32) : N.t =
  P.training_test x y
