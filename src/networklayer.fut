
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

module random = import "/futlib/random"
module rng_engine = random.minstd_rand0
module array  = import "/futlib/array"

module NetworkLayer (real: real) :
  { type t = real.t
    type layer [i] [j] = { biases: [j]t, weights: [j][i]t, rng: rng_engine.rng }
    type layer_modify [i] [j] = { biases: *[j]t, weights: *[j][i]t, rng: rng_engine.rng }
    val zero: t
    val one: t
    val matvecmul [n] [m] : (xss: [n][m]t) -> (ys: [m]t) -> [m]t
    val network_layer : rng_engine.rng -> (from: i32) -> (to: i32) -> layer [from] [to]
    val feedforward_layer [i] [j] : layer [i] [j] -> [i]t -> [j]t
    val rnd_permute 't [n] : rng_engine.rng -> [n]t -> (rng_engine.rng, [n]t)
    val sigmoid : (x:t) -> t
    val sigmoid_prime : (x:t) -> t
  } = {

  module Linalg = linalg(real)
  module ndist = random.normal_distribution real rng_engine

  module pair_radix_sort = mk_radix_sort {
    type t = (i32,i32)
    let num_bits = 32
    let get_bit (bit: i32) (x:i32,_:i32) = (x >> bit) & 1
  }

  type t = real.t

  let one = real.from_fraction 1 1
  let zero = real.floor (real.from_fraction 1 2)

  let dotprod [n] (xs: [n]t) (ys: [n]t): t =
    let zs = map (real.*) xs ys
    let a = zs[0]
    let zero = (a real.+one)real.*(a real.+one) real.- a real.*a real.- a real.- a real.- one  -- avoid fusion
    in reduce (real.+) zero zs

  let matvecmul [n] [m] (xss: [n][m]t) (ys: [m]t) =
    map (dotprod ys) xss

  let stddist : ndist.distribution = {mean=zero,stddev=one}

  -- The sigmoid function
  let sigmoid (x:t) =
    one real./ (one real.+ real.exp(real.negate x))

  -- Derivative of sigmoid
  let sigmoid_prime (x:t) =
    let s = sigmoid x
    in s real.* (one real.- s)

  -- Random numbers and random permutations
  type rng = rng_engine.rng

  let rand (rng:rng) (n:i32) : (rng,[n]i32) =
    let rngs = rng_engine.split_rng n rng
    let pairs = map (\rng -> rng_engine.rand rng) rngs
    let (rngs',a) = unzip pairs
    let a = map (i32.u32) a
    in (rng_engine.join_rng rngs', a)

  -- [rnd_perm n] returns an array of size n containing a random permutation of iota n.
  let rnd_perm (rng:rng) (n:i32) : (rng,[n]i32) =
    let (rng,a) = rand rng n
    let b = map (\x i -> (x,i)) a (iota n)
    let c = pair_radix_sort.radix_sort b
    let is = map (\(_,i) -> i) c
    in (rng,is)

  let rnd_permute 't [n] (rng:rng) (a:[n]t) : (rng,[n]t) =
    let (rng,is) = rnd_perm rng n
    in unsafe(rng,map (\i -> a[i]) is)

  let randn (rng:rng) (n:i32) : (rng,*[n]t) =
    let rngs = rng_engine.split_rng n rng
    let pairs = map (\rng -> ndist.rand stddist rng) rngs
    let (rngs',a) = unzip pairs
    in (rng_engine.join_rng rngs', a)

  let randn_bad (rng:rng) (n:i32) : (rng,*[n]t) =
    (rng,
     replicate n (zero))

   -- Network layer
   type layer [i] [j] = { biases: [j]t, weights: [j][i]t, rng: rng_engine.rng }
   type layer_modify [i] [j] = { biases: *[j]t, weights: *[j][i]t, rng: rng_engine.rng }

   let network_layer (rng:rng) (prev_sz:i32) (sz:i32) : layer [prev_sz] [sz] =
     let (rng,biases) = randn rng sz
     let (rng,weights_flat) = randn rng (sz*prev_sz)
     let weights_flat' = map (\w -> w real./ real.sqrt(real.from_fraction prev_sz 1)) weights_flat
     let weights = reshape (sz,prev_sz) weights_flat'
     in {rng = rng, biases = biases, weights = weights}

   let feedforward_layer [i] [j] (layer: layer [i] [j]) (arg:[i]t): [j]t =
     let t = matvecmul layer.weights arg
     in map (\b t -> sigmoid (t real.+ b)) layer.biases t

}
