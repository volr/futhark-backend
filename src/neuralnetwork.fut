
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

module random = import "/futlib/random"
module rng_engine = random.minstd_rand0
module array  = import "/futlib/array"

module generic_predict (real: real)
  : { type t = real.t
      val run : ([]t,[]i32,[]t,[]i32) -> t
    } =
{
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

  let dotprod [n] (xs: [n]real.t) (ys: [n]real.t): real.t =
    let zs = map (real.*) xs ys
    let a = zs[0]
    let zero = (a real.+one)real.*(a real.+one) real.- a real.*a real.- a real.- a real.- one  -- avoid fusion
    in reduce (real.+) zero zs

  let matvecmul [n] [m] (xss: [n][m]real.t) (ys: [m]real.t) =
    map (dotprod ys) xss

  let stddist : ndist.distribution = {mean=zero,stddev=one}

  -- The sigmoid function
  let sigmoid (x:real.t) =
    one real./ (one real.+ real.exp(real.negate x))

  -- Derivative of sigmoid
  let sigmoid_prime (x:real.t) =
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

  let randn (rng:rng) (n:i32) : (rng,*[n]real.t) =
    let rngs = rng_engine.split_rng n rng
    let pairs = map (\rng -> ndist.rand stddist rng) rngs
    let (rngs',a) = unzip pairs
    in (rng_engine.join_rng rngs', a)

  let randn_bad (rng:rng) (n:i32) : (rng,*[n]real.t) =
    (rng,
     replicate n (zero))

  -- Network layers

  type layer [i] [j] = ([j]real.t, [j][i]real.t)
  type layeru [i] [j] = (*[j]real.t, *[j][i]real.t)

  let network_layer (rng:rng) (prev_sz:i32) (sz:i32) : (rng,(*[sz]real.t, *[sz][prev_sz]real.t)) =
    let (rng,biases) = randn rng sz
    let (rng,weights_flat) = randn rng (sz*prev_sz)
    let weights_flat' = map (\w -> w real./ real.sqrt(real.from_fraction prev_sz 1)) weights_flat
    let weights = reshape (sz,prev_sz) weights_flat'
    in (rng,(biases,weights))

  -- Initialise a network given a configuration (a vector of neuron
  -- numbers, one number for each layer).

  type network3 [i] [j] [k] = (layer [i] [j], layer [j] [k])
  type network3u [i] [j] [k] = (layeru [i] [j], layeru [j] [k])

  let network3 (rng:rng) (sz1:i32) (sz2:i32) (sz3:i32) : (rng,network3u [sz1] [sz2] [sz3]) =
    let (rng,layer2) = network_layer rng sz1 sz2
    let (rng,layer3) = network_layer rng sz2 sz3
    in (rng,(layer2,layer3))

  let feedforward_layer [i] [j] (b:[j]real.t, w:[j][i]real.t) (arg:[i]real.t) : [j]real.t =
    let t = matvecmul w arg
    in map (\b t -> sigmoid (t real.+ b)) b t

  -- [(B W) feedforward3 a] returns the output of the network (B,W) given
  -- the input a.
  let feedforward3 [i] [j] [k] (layer2:layer[i][j],layer3:layer[j][k]) (a:[i]real.t) : [k]real.t =
    let a = feedforward_layer layer2 a
    let a = feedforward_layer layer3 a
    in a

  let cost_derivative [n] (output_activations:[n]real.t) (y:[n]real.t) : [n]real.t =
    map (real.-) output_activations y

  let sub_network [i][j][k] (nabla_factor: real.t) (weight_factor: real.t)
                            (network:network3[i][j][k]) (nabla:network3[i][j][k]) =
    let ((b2,w2),(b3,w3)) = network
    let ((b2n,w2n),(b3n,w3n)) = nabla
    let sub_bias (b:real.t) (nb:real.t) : real.t =
      b real.- (nabla_factor real.* nb)
    let sub_weight (w:real.t) (nw:real.t) : real.t =
      (weight_factor real.* w) real.- (nabla_factor real.* nw)
    let b2' = map sub_bias b2 b2n
    let w2' = map (\x y -> map sub_weight x y) w2 w2n
    let b3' = map sub_bias b3 b3n
    let w3' = map (\x y -> map sub_weight x y) w3 w3n
    in ((b2',w2'),(b3',w3'))

  let outer_prod [m][n] (a:[m]real.t) (b:[n]real.t) : *[m][n]real.t =
    map (\x -> map (\y -> x real.* y) b) a

  let backprop [i] [j] [k] (network:network3[i][j][k])
                           (x:[i]real.t,y:[k]real.t) : network3u[i][j][k] =
    -- Return a nabla (a tuple ``(nabla_b, nabla_w)``) for each (non-input)
    -- layer, which, together, represent the gradient for the cost function C_x.
    -- Feedforward
    let ((b2,w2),(b3,w3)) = network
    let activation1 = x
    let z2 = map (real.+) (matvecmul w2 activation1) b2
    let activation2 = map sigmoid z2
    let z3 = map (real.+) (matvecmul w3 activation2) b3
    let activation3 = map sigmoid z3
    -- Backward pass
    let delta3 = map (real.*) (cost_derivative activation3 y)
                     (map sigmoid_prime z3)
    let nabla_b3 = delta3
    let nabla_w3 = outer_prod delta3 activation2
    let sp = map sigmoid_prime z2
    let delta2 = map (real.*) (Linalg.matvecmul_row (array.transpose w3) delta3) sp
    let nabla_b2 = delta2
    let nabla_w2 = outer_prod delta2 activation1
    let nabla2 = (nabla_b2,nabla_w2)
    let nabla3 = (nabla_b3,nabla_w3)
    in (nabla2,nabla3)

  let seq_sum [n] (a:[n]real.t) : real.t =
    loop s = zero for i < n do s real.+ a[i]

  let layer_sum [n] [i] [j] (a: [n](layer[i][j])) : layer[i][j] =
    -- For layer_sum, we use a sequential inner loop to sum over the
    -- appropriate dimensions; an alternative is to use rearrange and
    -- transposes, which may introduce an overhead...
    let (bs,ws) = unzip a
    --let b = map (\xs -> seq_sum xs) (array.transpose bs)
    let b = map (\jj -> loop s = zero for k < n do s real.+ bs[k,jj]) (iota j)
    --let w = map (\rs -> map (\xs -> seq_sum xs) rs) (rearrange (1,2,0) ws)   -- i,j,n
    let w = map (\jj -> map (\ii -> loop s = zero for k < n do s real.+ ws[k,jj,ii]) (iota i)) (iota j)
    in (b,w)

  let layer_sum_par [n] [i] [j] (a: [n](layer[i][j])) : layer[i][j] =
    -- Alternative parallel version of layer_sum
    let (bs,ws) = unzip a
    let b = map (\xs -> reduce (real.+) zero xs) (array.transpose bs)
    let w = map (\rs -> map (\xs -> reduce (real.+) zero xs) rs) (rearrange (1,2,0) ws)   -- i,j,n
    in (b,w)

  let network3_sum [n] [i] [j] [k] (a: [n](network3[i][j][k])) : network3[i][j][k] =
    let (ls2,ls3) = unzip a
    in (layer_sum ls2, layer_sum ls3)

  let update_mini_batch [n] [i] [j] [k] (eta:real.t)
                                        (lmbda:real.t)
                                        (training_len: i32)
                                        (network:network3[i][j][k])
                                        (mini_batch:[n]([i]real.t,[k]real.t)) : network3[i][j][k] =
    -- Update the network's weights and biases by applying
    -- gradient descent using backpropagation to a single mini batch.
    -- The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    -- is the learning rate.
    let delta_nabla = map (\d -> backprop network d) mini_batch
    let nabla = network3_sum delta_nabla
    let nabla_factor = eta real./ (real.from_fraction n 1)
    let weight_factor = (one) real.-
                        (eta real.* (lmbda real./ (real.from_fraction training_len 1)))
    in sub_network nabla_factor weight_factor network nabla

  let sgd [i] [j] [k] [n] (rng: rng,
                           network: network3[i][j][k],
                           training_data: [n]([i]real.t,[k]real.t),
                           epochs:i32,
                           mini_batch_size:i32,
                           eta:real.t,
                           lmbda: real.t) : network3[i][j][k] =
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
        let (rng,training_data) = rnd_permute rng training_data
        let (a,b) = unzip training_data
        let a = reshape (batches,mini_batch_size,i) a
        let b = reshape (batches,mini_batch_size,k) b
        let network =
          loop network for x < batches do
            update_mini_batch eta lmbda n network (zip a[x] b[x])
        in (rng,network)
    in network

  let convert_digit (d:i32) : [10]real.t =
    let a = replicate 10 zero
    in unsafe(a with [d] <- one)

  let predict (a:[10]real.t) : i32 =
    let (_,i) = reduce (\(a,i) (b,j) -> if a real.> b then (a,i) else (b,j))
                       (a[9],9)
                       (zip (a[:9]) (iota 9))
    in i

  let run [m] [n] [m2] [n2] (training_imgs:[m]real.t,
                             training_results:[n]i32,
                             test_imgs:[m2]real.t,
                             test_results:[n2]i32) : real.t =
    let rng = rng_engine.rng_from_seed [0]
    let epochs = 10
    let lmbda = zero
    let mini_batch_size = 20
    let eta = real.from_fraction 1 2
    let imgs = reshape (n, 28*28) training_imgs
    let data = map (\img d -> (img,convert_digit d)) imgs training_results
    -- split rng
    let (rng,n0) = network3 rng (28*28) 30 10
    let n = sgd (rng, n0, data, epochs, mini_batch_size, eta, lmbda)
    let t_imgs = reshape (n2, 28*28) test_imgs
    let predictions = map (\img -> predict(feedforward3 n img)) t_imgs
    let cmps = map (\p r -> i32.bool(p==r)) predictions test_results
    in (real.from_fraction 100 1) real.* (real.from_fraction(reduce (+) 0 cmps) 1) real./ (real.from_fraction n2 1)
}

module predict = digit_predict(f32) -- f32 or f64

let main (training_imgs:[]predict.t,
          training_results:[]i32,
          test_imgs:[]predict.t,
          test_results:[]i32) : predict.t =
  predict.run (training_imgs,
               training_results,
               test_imgs,
               test_results)
