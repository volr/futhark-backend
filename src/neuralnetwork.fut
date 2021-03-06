
---------------------------------------
-- Module for layered perceptrons
-- Author: Jens Egholm <jensegholm@protonmail.com>
-- Heavily inspired by Martin Elsmans implementation:
--    https://github.com/melsman/neural-networks-and-deep-learning
-- License: GPLv3
---------------------------------------

import "/futlib/linalg"
import "/futlib/math"
import "/futlib/radix_sort"

import "networklayer"

module random = import "/futlib/random"
module rng_engine = random.minstd_rand0
module array  = import "/futlib/array"

type rng = rng_engine.rng

module type NeuralNetwork = {
  type t
  type input
  type output
  val run [m] [n] [l] : ([n][m]t, [n]i32, [l][m]t, [l]i32) -> t
}

module type Config3 = {
  val size1: i32
  val size2: i32
  val output: i32
  val learning_rate: f64
}

module type Config4 = {
  include Config3
  val size3: i32
}

module Network3 (real: real) (N: Config3) : NeuralNetwork with t = real.t = {
  module Linalg = linalg(real)
  module L = NetworkLayer(real)

  type t = L.t
  type input = [N.size1]t
  type output = [N.output]t

  type network3 [l] [m] [n] = (L.layer [l] [m], L.layer [m] [n])
  type network3u [l] [m] [n] = (L.layer_modify [l] [m], L.layer_modify [m] [n])

  let layer_sum [n] [i] [j] (a: [n](L.layer[i][j])) : L.layer[i][j] =
    -- For layer_sum, we use a sequential inner loop to sum over the
    -- appropriate dimensions; an alternative is to use rearrange and
    -- transposes, which may introduce an overhead...
    let bs = map (\l -> l.biases) a
    let ws = map (\l -> l.weights) a
    let b = map (\jj -> loop s = L.zero for k < n do s real.+ bs[k,jj]) (iota j)
    let w = map (\jj -> map (\ii -> loop s = L.zero for k < n do s real.+ ws[k,jj,ii]) (iota i)) (iota j)
    in {biases = b, weights = w, rng = a[-1].rng}

  let feedforward [i] [j] [k] (network: network3[i][j][k])(i: input) : output =
    let (layer1, layer2) = network
    let i = L.feedforward_layer layer1 i
    let i = L.feedforward_layer layer2 i
    in i

  let network3_sum [n] [i] [j] [k] (a: [n](network3[i][j][k])) : network3[i][j][k] =
    let (ls2,ls3) = unzip a
    in (layer_sum ls2, layer_sum ls3)

  let cost_derivative [n] (output_activations:[n]t) (y:[n]t) : [n]t =
    map (real.-) output_activations y

  let outer_prod [m][n] (a:[m]t) (b:[n]t) : *[m][n]t =
    map (\x -> map (\y -> x real.* y) b) a

  let backprop [i] [j] [k] (network: network3[i][j][k]) (x:[i]t,y:[k]t) : network3u[i][][k] =
    -- Return a nabla (a tuple ``(nabla_b, nabla_w)``) for each (non-input)
    -- layer, which, together, represent the gradient for the cost function C_x.
    -- Feedforward
    let ({biases = b2, weights = w2, rng = r2}, {biases = b3, weights = w3, rng = r3}) = network
    let activation1 = x
    let z2 = map (real.+) (L.matvecmul w2 activation1) b2
    let activation2 = map L.sigmoid z2
    let z3 = map (real.+) (L.matvecmul w3 activation2) b3
    let activation3 = map L.sigmoid z3
    -- Backward pass
    let delta3 = map (real.*) (cost_derivative activation3 y)
                     (map L.sigmoid_prime z3)
    let nabla_b3 = delta3
    let nabla_w3 = outer_prod delta3 activation2
    let sp = map L.sigmoid_prime z2
    let delta2 = map (real.*) (Linalg.matvecmul_row (array.transpose w3) delta3) sp
    let nabla_b2 = delta2
    let nabla_w2 = outer_prod delta2 activation1
    let nabla2 = {biases = nabla_b2, weights = nabla_w2, rng = r2}
    let nabla3 = {biases = nabla_b3, weights = nabla_w3, rng = r3}
    in (nabla2,nabla3)

  let sub_network [i][j][k] (nabla_factor: t) (weight_factor: t)
                            (network: network3[i][j][k])(nabla:network3[i][j][k]) =
    let ({biases = b2, weights = w2, rng = r2},{biases = b3, weights = w3, rng = r3}) = network
    let ({biases = b2n, weights = w2n, rng = _}, {biases = b3n, weights = w3n, rng = _}) = nabla
    let sub_bias (b:t) (nb:t) : t =
      b real.- (nabla_factor real.* nb)
    let sub_weight (w:t) (nw:t) : t =
      (weight_factor real.* w) real.- (nabla_factor real.* nw)
    let b2' = map sub_bias b2 b2n
    let w2' = map (\x y -> map sub_weight x y) w2 w2n
    let b3' = map sub_bias b3 b3n
    let w3' = map (\x y -> map sub_weight x y) w3 w3n
    in ({biases = b2', weights = w2', rng = r2},{biases = b3', weights = w3', rng = r3})

  let update_mini_batch [n] [i] [j] [k] (eta:t)
                                        (lmbda:t)
                                        (training_len: i32)
                                        (network: network3[i][j][k])
                                        (batch:[n]([i]t,[k]t)) : network3u[i][][k] =
    -- Update the network's weights and biases by applying
    -- gradient descent using backpropagation to a single mini batch.
    -- The ``batch`` is a list of tuples ``(x, y)``, and ``eta``
    -- is the learning rate.
    let delta_nabla = map (\d -> backprop network d) batch
    let nabla = network3_sum delta_nabla
    let nabla_factor = eta real./ (real.from_fraction n 1)
    let weight_factor = (L.one) real.-
                        (eta real.* (lmbda real./ (real.from_fraction training_len 1)))
    in sub_network nabla_factor weight_factor network nabla

    let sgd [i] [j] [k] [n] (rng: rng,
                             network: network3[i][j][k],
                             training_data: [n]([i]t,[k]t),
                             epochs:i32,
                             mini_batch_size:i32,
                             eta:t,
                             lmbda: t) : network3[i][j][k] =
      -- Train the neural network using mini-batch stochastic
      -- gradient descent.  The ``training_data`` is a list of tuples
      -- ``(x, y)`` representing the training inputs and the desired
      -- outputs.  The other non-optional parameters are
      -- self-explanatory.
      let batches = n / mini_batch_size
      let n = batches * mini_batch_size
      let training_data = training_data[:n]
      let (_,network) =
        loop (rng,network) for j < epochs do
          let (rng,training_data) = L.rnd_permute rng training_data
          let (a,b) = unzip training_data
          let a = reshape (batches,mini_batch_size,i) a
          let b = reshape (batches,mini_batch_size,k) b
          let network =
            loop network for x < batches do
              update_mini_batch eta lmbda n network (zip a[x] b[x])
          in (rng,network)
      in network

  let convert (d:i32) : input =
    let a = replicate N.size1 L.zero
    in unsafe(a with [d] <- L.one)

  let predict (a:output) : i32 =
    let n = N.output - 1
    let (_,i) = reduce (\(a,i) (b,j) -> if a real.> b then (a,i) else (b,j))
                       (a[n],n)
                       (zip (a[:n]) (iota (n)))
    in i

  let run [m] [n] [l] (training_x:[n][m]t,
                       training_y:[n]i32,
                       test_x:[l][m]t,
                       test_y:[l]i32) : t =
    let rng = rng_engine.rng_from_seed [0]
    let epochs = 10
    let lmbda = L.zero
    let mini_batch_size = 20

    let data : []([]t, []t) = zip training_x (map convert training_y)
    -- split rng
    let layer1 = L.network_layer rng N.size1 N.size2
    let layer2 = L.network_layer layer1.rng N.size2 N.output
    -- train
    let n = sgd (rng, (layer1, layer2), data, epochs, mini_batch_size, real.f64 N.learning_rate, lmbda)
    -- test
    let predictions = map (\d -> predict(feedforward n d)) test_x
    let cmps = map (\p r -> i32.bool(p==r)) predictions test_y
    in (real.from_fraction 100 1) real.* (real.from_fraction(reduce (+) 0 cmps) 1) real./ (real.from_fraction l 1)
}

module Network4 (real: real) (N: Config4) : NeuralNetwork with t = real.t = {

  module Linalg = linalg(real)
  module L = NetworkLayer(real)
  type t = L.t

  type input = [N.size1]t
  type output = [N.output]t

  type network4 [l] [m] [n] [o] = (L.layer [l] [m], L.layer [m] [n], L.layer [n] [o])
  type network4u [l] [m] [n] [o] = (L.layer_modify [l] [m], L.layer_modify [m] [n], L.layer_modify [n] [o])

  let layer_sum [n] [i] [j] (a: [n](L.layer[i][j])) : L.layer[i][j] =
    -- For layer_sum, we use a sequential inner loop to sum over the
    -- appropriate dimensions; an alternative is to use rearrange and
    -- transposes, which may introduce an overhead...
    let bs = map (\l -> l.biases) a
    let ws = map (\l -> l.weights) a
    let b = map (\jj -> loop s = L.zero for k < n do s real.+ bs[k,jj]) (iota j)
    let w = map (\jj -> map (\ii -> loop s = L.zero for k < n do s real.+ ws[k,jj,ii]) (iota i)) (iota j)
    in {biases = b, weights = w, rng = a[-1].rng}

  let feedforward [i] [j] [k] [l](network: network4[i][j][k][l])(i: input) : output =
    let (layer1, layer2, layer3) = network
    let i = L.feedforward_layer layer1 i
    let i = L.feedforward_layer layer2 i
    let i = L.feedforward_layer layer3 i
    in i

  let network4_sum [n] [i] [j] [k] [l] (a: [n](network4[i][j][k][l])) : network4[i][j][k][l] =
    let (ls2,ls3,ls4) = unzip a
    in (layer_sum ls2, layer_sum ls3, layer_sum ls4)

  let cost_derivative [n] (output_activations:[n]t) (y:[n]t) : [n]t =
    map (real.-) output_activations y

  let outer_prod [m][n] (a:[m]t) (b:[n]t) : *[m][n]t =
    map (\x -> map (\y -> x real.* y) b) a

  let backprop [i] [j] [k] [l] (network: network4[i][j][k][l]) (x:[i]t,y:[l]t) : network4u[i][j][k][l] =
    -- Return a nabla (a tuple ``(nabla_b, nabla_w)``) for each (non-input)
    -- layer, which, together, represent the gradient for the cost function C_x.
    -- Feedforward
    let ({biases = b2, weights = w2, rng = r2},
         {biases = b3, weights = w3, rng = r3},
         {biases = b4, weights = w4, rng = r4}) = network
    let activation1 = x
    let z2 = map (real.+) (L.matvecmul w2 activation1) b2
    let activation2 = map L.sigmoid z2
    let z3 = map (real.+) (L.matvecmul w3 activation2) b3
    let activation3 = map L.sigmoid z3
    let z4 = map (real.+) (L.matvecmul w4 activation3) b4
    let activation4 = map L.sigmoid z4
    -- Backward pass
    let delta4 = map (real.*) (cost_derivative activation4 y)
                     (map L.sigmoid_prime z4)
    let nabla_b4 = delta4
    let nabla_w4 = outer_prod delta4 activation3
    let sp = map L.sigmoid_prime z3
    let delta3 = map (real.*) (Linalg.matvecmul_row (array.transpose w4) delta4) sp
    let nabla_b3 = delta3
    let nabla_w3 = outer_prod delta3 activation2
    let sp = map L.sigmoid_prime z2
    let delta2 = map (real.*) (Linalg.matvecmul_row (array.transpose w3) delta3) sp
    let nabla_b2 = delta2
    let nabla_w2 = outer_prod delta2 activation1
    let nabla2 = {biases = nabla_b2, weights = nabla_w2, rng = r2}
    let nabla3 = {biases = nabla_b3, weights = nabla_w3, rng = r3}
    let nabla4 = {biases = nabla_b4, weights = nabla_w4, rng = r4}
    in (nabla2,nabla3,nabla4)

  let sub_network [i][j][k][l] (nabla_factor: t) (weight_factor: t)
                            (network: network4[i][j][k][l])(nabla:network4[i][j][k][l]) =
    let ({biases = b2, weights = w2, rng = r2},{biases = b3, weights = w3, rng = r3},
         {biases = b4, weights = w4, rng = r4}) = network
    let ({biases = b2n, weights = w2n, rng = _},
         {biases = b3n, weights = w3n, rng = _},
         {biases = b4n, weights = w4n, rng = _}) = nabla
    let sub_bias (b:t) (nb:t) : t =
      b real.- (nabla_factor real.* nb)
    let sub_weight (w:t) (nw:t) : t =
      (weight_factor real.* w) real.- (nabla_factor real.* nw)
    let b2' = map sub_bias b2 b2n
    let w2' = map (\x y -> map sub_weight x y) w2 w2n
    let b3' = map sub_bias b3 b3n
    let w3' = map (\x y -> map sub_weight x y) w3 w3n
    let b4' = map sub_bias b4 b4n
    let w4' = map (\x y -> map sub_weight x y) w4 w4n
    in ({biases = b2', weights = w2', rng = r2},
        {biases = b3', weights = w3', rng = r3},
        {biases = b4', weights = w4', rng = r4})

  let update_mini_batch [n] [i] [j] [k] [l]
      (eta:t) (lmbda:t) (training_len: i32) (network: network4[i][j][k][l])
      (batch:[n]([i]t,[l]t)) : network4u[i][j][k][l] =
    -- Update the network's weights and biases by applying
    -- gradient descent using backpropagation to a single mini batch.
    -- The ``batch`` is a list of tuples ``(x, y)``, and ``eta``
    -- is the learning rate.
    let delta_nabla = map (\d -> backprop network d) batch
    let nabla = network4_sum delta_nabla
    let nabla_factor = eta real./ (real.from_fraction n 1)
    let weight_factor = (L.one) real.-
                        (eta real.* (lmbda real./ (real.from_fraction training_len 1)))
    in sub_network nabla_factor weight_factor network nabla

    let sgd [i] [j] [k] [l] [n] (rng: rng, network: network4[i][j][k][l],
            training_data: [n]([i]t,[l]t), epochs:i32,
            mini_batch_size:i32, eta:t, lmbda: t) : network4[i][j][k][l] =
      -- Train the neural network using mini-batch stochastic
      -- gradient descent.  The ``training_data`` is a list of tuples
      -- ``(x, y)`` representing the training inputs and the desired
      -- outputs.  The other non-optional parameters are
      -- self-explanatory.
      let batches = n / mini_batch_size
      let n = batches * mini_batch_size
      let training_data = training_data[:n]
      let (_,network) =
        loop (rng,network) for j < epochs do
          let (rng,training_data) = L.rnd_permute rng training_data
          let (a,b) = unzip training_data
          let a = reshape (batches,mini_batch_size,i) a
          let b = reshape (batches,mini_batch_size,l) b
          let network =
            loop network for x < batches do
              update_mini_batch eta lmbda n network (zip a[x] b[x])
          in (rng,network)
      in network

  let convert (d:i32) : input =
    let a = replicate N.size1 L.zero
    in unsafe(a with [d] <- L.one)

  let predict (a:output) : i32 =
    let n = N.output - 1
    let (_,i) = reduce (\(a,i) (b,j) -> if a real.> b then (a,i) else (b,j))
                       (a[n],n)
                       (zip (a[:n]) (iota (n)))
    in i

  let run [n] [m] [l] (training_x:[n][m]t,
                       training_y:[n]i32,
                       test_x:[l][m]t,
                       test_y:[l]i32) : t =
    let rng : rng = rng_engine.rng_from_seed [0]
    let epochs: i32 = 10
    let lmbda: t = L.zero
    let mini_batch_size: i32 = 20

    let data : [n]([N.size1]t, [N.output]t) = zip training_x (map convert training_y)
    -- split rng
    let layer1 = L.network_layer rng N.size1 N.size2
    let layer2 = L.network_layer layer1.rng N.size2 N.size3
    let layer3 = L.network_layer layer2.rng N.size3 N.output
    let network : network4 [N.size1] [N.size2] [N.size3] [N.output] = (layer1, layer2, layer3)
    -- train
    let n = sgd (rng, network, data, epochs, mini_batch_size, real.f64 N.learning_rate, lmbda)
    -- test
    let predictions = map (\d -> predict(feedforward n d)) test_x
    let cmps = map (\p r -> i32.bool(p==r)) predictions test_y
    in (real.i32 100) real.* (real.i32 (reduce (+) 0 cmps)) real./ (real.from_fraction l 1)
}
