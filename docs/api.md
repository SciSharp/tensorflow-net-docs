# API

## Functions

| Name        | Description                               | Has Test Case | Has Completed |
| ----------- | ----------------------------------------- | ------------- | ------------- |
| [`Assert(...)`](https://www.tensorflow.org/api_docs/python/tf/debugging/Assert) | : Asserts that the given condition is true. | | |
| [`abs(...)`](https://www.tensorflow.org/api_docs/python/tf/math/abs) | : Computes the absolute value of a tensor. | | |
| [`acos(...)`](https://www.tensorflow.org/api_docs/python/tf/math/acos) | : Computes acos of x element-wise. | | |
| [`acosh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/acosh) | : Computes inverse hyperbolic cosine of x element-wise. | | |
| [`add(...)`](https://www.tensorflow.org/api_docs/python/tf/math/add) | : Returns x + y element-wise. | | |
| [`add_n(...)`](https://www.tensorflow.org/api_docs/python/tf/math/add_n) | : Adds all input tensors element-wise. | | |
| [`argmax(...)`](https://www.tensorflow.org/api_docs/python/tf/math/argmax) | : Returns the index with the largest value across axes of a tensor. | | |
| [`argmin(...)`](https://www.tensorflow.org/api_docs/python/tf/math/argmin) | : Returns the index with the smallest value across axes of a tensor. | | |
| [`argsort(...)`](https://www.tensorflow.org/api_docs/python/tf/argsort) | : Returns the indices of a tensor that give its sorted order along an axis. | | |
| [`as_dtype(...)`](https://www.tensorflow.org/api_docs/python/tf/dtypes/as_dtype) | : Converts the given `type_value` to a `DType`. | | |
| [`as_string(...)`](https://www.tensorflow.org/api_docs/python/tf/strings/as_string) | : Converts each entry in the given tensor to strings. | | |
| [`asin(...)`](https://www.tensorflow.org/api_docs/python/tf/math/asin) | : Computes the trignometric inverse sine of x element-wise. | | |
| [`asinh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/asinh) | : Computes inverse hyperbolic sine of x element-wise. | | |
| [`assert_equal(...)`](https://www.tensorflow.org/api_docs/python/tf/debugging/assert_equal) | : Assert the condition `x == y` holds element-wise. | | |
| [`assert_greater(...)`](https://www.tensorflow.org/api_docs/python/tf/debugging/assert_greater) | : Assert the condition `x > y` holds element-wise. | | |
| [`assert_less(...)`](https://www.tensorflow.org/api_docs/python/tf/debugging/assert_less) | : Assert the condition `x < y` holds element-wise. | | |
| [`assert_rank(...)`](https://www.tensorflow.org/api_docs/python/tf/debugging/assert_rank) | : Assert that `x` has rank equal to `rank`. | | |
| [`atan(...)`](https://www.tensorflow.org/api_docs/python/tf/math/atan) | : Computes the trignometric inverse tangent of x element-wise. | | |
| [`atan2(...)`](https://www.tensorflow.org/api_docs/python/tf/math/atan2) | : Computes arctangent of `y/x` element-wise, respecting signs of the arguments. | | |
| [`atanh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/atanh) | : Computes inverse hyperbolic tangent of x element-wise. | | |
| [`batch_to_space(...)`](https://www.tensorflow.org/api_docs/python/tf/batch_to_space) | : BatchToSpace for N-D tensors of type T. | | |
| [`bitcast(...)`](https://www.tensorflow.org/api_docs/python/tf/bitcast) | : Bitcasts a tensor from one type to another without copying data. | | |
| [`boolean_mask(...)`](https://www.tensorflow.org/api_docs/python/tf/boolean_mask) | : Apply boolean mask to tensor. | | |
| [`broadcast_dynamic_shape(...)`](https://www.tensorflow.org/api_docs/python/tf/broadcast_dynamic_shape) | : Computes the shape of a broadcast given symbolic shapes. | | |
| [`broadcast_static_shape(...)`](https://www.tensorflow.org/api_docs/python/tf/broadcast_static_shape) | : Computes the shape of a broadcast given known shapes. | | |
| [`broadcast_to(...)`](https://www.tensorflow.org/api_docs/python/tf/broadcast_to) | : Broadcast an array for a compatible shape. | | |
| [`case(...)`](https://www.tensorflow.org/api_docs/python/tf/case) | : Create a case operation. | | |
| [`cast(...)`](https://www.tensorflow.org/api_docs/python/tf/cast) | : Casts a tensor to a new type. | | |
| [`clip_by_global_norm(...)`](https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm) | : Clips values of multiple tensors by the ratio of the sum of their norms. | | |
| [`clip_by_norm(...)`](https://www.tensorflow.org/api_docs/python/tf/clip_by_norm) | : Clips tensor values to a maximum L2-norm. | | |
| [`clip_by_value(...)`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value) | : Clips tensor values to a specified min and max. | | |
| [`complex(...)`](https://www.tensorflow.org/api_docs/python/tf/dtypes/complex) | : Converts two real numbers to a complex number. | | |
| [`concat(...)`](https://www.tensorflow.org/api_docs/python/tf/concat) | : Concatenates tensors along one dimension. | | |
| [`cond(...)`](https://www.tensorflow.org/api_docs/python/tf/cond) | : Return `true_fn()` if the predicate `pred` is true else `false_fn()`. | | |
| [`constant(...)`](https://www.tensorflow.org/api_docs/python/tf/constant) | : Creates a constant tensor from a tensor-like object. | | |
| [`control_dependencies(...)`](https://www.tensorflow.org/api_docs/python/tf/control_dependencies) | : Wrapper for [`Graph.control_dependencies()`](https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies) using the default graph. | | |
| [`convert_to_tensor(...)`](https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor) | : Converts the given `value` to a `Tensor`. | | |
| [`cos(...)`](https://www.tensorflow.org/api_docs/python/tf/math/cos) | : Computes cos of x element-wise. | | |
| [`cosh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/cosh) | : Computes hyperbolic cosine of x element-wise. | | |
| [`cumsum(...)`](https://www.tensorflow.org/api_docs/python/tf/math/cumsum) | : Compute the cumulative sum of the tensor `x` along `axis`. | | |
| [`custom_gradient(...)`](https://www.tensorflow.org/api_docs/python/tf/custom_gradient) | : Decorator to define a function with a custom gradient. | | |
| [`device(...)`](https://www.tensorflow.org/api_docs/python/tf/device) | : Specifies the device for ops created/executed in this context. | | |
| [`divide(...)`](https://www.tensorflow.org/api_docs/python/tf/math/divide) | : Computes Python style division of `x` by `y`. | | |
| [`dynamic_partition(...)`](https://www.tensorflow.org/api_docs/python/tf/dynamic_partition) | : Partitions `data` into `num_partitions` tensors using indices from `partitions`. | | |
| [`dynamic_stitch(...)`](https://www.tensorflow.org/api_docs/python/tf/dynamic_stitch) | : Interleave the values from the `data` tensors into a single tensor. | | |
| [`edit_distance(...)`](https://www.tensorflow.org/api_docs/python/tf/edit_distance) | : Computes the Levenshtein distance between sequences. | | |
| [`eig(...)`](https://www.tensorflow.org/api_docs/python/tf/linalg/eig) | : Computes the eigen decomposition of a batch of matrices. | | |
| [`eigvals(...)`](https://www.tensorflow.org/api_docs/python/tf/linalg/eigvals) | : Computes the eigenvalues of one or more matrices. | | |
| [`einsum(...)`](https://www.tensorflow.org/api_docs/python/tf/einsum) | : Tensor contraction over specified indices and outer product. | | |
| [`ensure_shape(...)`](https://www.tensorflow.org/api_docs/python/tf/ensure_shape) | : Updates the shape of a tensor and checks at runtime that the shape holds. | | |
| [`equal(...)`](https://www.tensorflow.org/api_docs/python/tf/math/equal) | : Returns the truth value of (x == y) element-wise. | | |
| [`executing_eagerly(...)`](https://www.tensorflow.org/api_docs/python/tf/executing_eagerly) | : Checks whether the current thread has eager execution enabled. | | |
| [`exp(...)`](https://www.tensorflow.org/api_docs/python/tf/math/exp) | : Computes exponential of x element-wise. \(y = e^x\). | | |
| [`expand_dims(...)`](https://www.tensorflow.org/api_docs/python/tf/expand_dims) | : Returns a tensor with a length 1 axis inserted at index `axis`. | | |
| [`extract_volume_patches(...)`](https://www.tensorflow.org/api_docs/python/tf/extract_volume_patches) | : Extract `patches` from `input` and put them in the `"depth"` output dimension. 3D extension of `extract_image_patches`. | | |
| [`eye(...)`](https://www.tensorflow.org/api_docs/python/tf/eye) | : Construct an identity matrix, or a batch of matrices. | | |
| [`fill(...)`](https://www.tensorflow.org/api_docs/python/tf/fill) | : Creates a tensor filled with a scalar value. | | |
| [`fingerprint(...)`](https://www.tensorflow.org/api_docs/python/tf/fingerprint) | : Generates fingerprint values. | | |
| [`floor(...)`](https://www.tensorflow.org/api_docs/python/tf/math/floor) | : Returns element-wise largest integer not greater than x. | | |
| [`foldl(...)`](https://www.tensorflow.org/api_docs/python/tf/foldl) | : foldl on the list of tensors unpacked from `elems` on dimension 0\. (deprecated argument values) | | |
| [`foldr(...)`](https://www.tensorflow.org/api_docs/python/tf/foldr) | : foldr on the list of tensors unpacked from `elems` on dimension 0\. (deprecated argument values) | | |
| [`function(...)`](https://www.tensorflow.org/api_docs/python/tf/function) | : Compiles a function into a callable TensorFlow graph. | | |
| [`gather(...)`](https://www.tensorflow.org/api_docs/python/tf/gather) | : Gather slices from params axis `axis` according to indices. | | |
| [`gather_nd(...)`](https://www.tensorflow.org/api_docs/python/tf/gather_nd) | : Gather slices from `params` into a Tensor with shape specified by `indices`. | | |
| [`get_logger(...)`](https://www.tensorflow.org/api_docs/python/tf/get_logger) | : Return TF logger instance. | | |
| [`get_static_value(...)`](https://www.tensorflow.org/api_docs/python/tf/get_static_value) | : Returns the constant value of the given tensor, if efficiently calculable. | | |
| [`grad_pass_through(...)`](https://www.tensorflow.org/api_docs/python/tf/grad_pass_through) | : Creates a grad-pass-through op with the forward behavior provided in f. | | |
| [`gradients(...)`](https://www.tensorflow.org/api_docs/python/tf/gradients) | : Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`. | | |
| [`greater(...)`](https://www.tensorflow.org/api_docs/python/tf/math/greater) | : Returns the truth value of (x > y) element-wise. | | |
| [`greater_equal(...)`](https://www.tensorflow.org/api_docs/python/tf/math/greater_equal) | : Returns the truth value of (x >= y) element-wise. | | |
| [`group(...)`](https://www.tensorflow.org/api_docs/python/tf/group) | : Create an op that groups multiple operations. | | |
| [`guarantee_const(...)`](https://www.tensorflow.org/api_docs/python/tf/guarantee_const) | : Gives a guarantee to the TF runtime that the input tensor is a constant. | | |
| [`hessians(...)`](https://www.tensorflow.org/api_docs/python/tf/hessians) | : Constructs the Hessian of sum of `ys` with respect to `x` in `xs`. | | |
| [`histogram_fixed_width(...)`](https://www.tensorflow.org/api_docs/python/tf/histogram_fixed_width) | : Return histogram of values. | | |
| [`histogram_fixed_width_bins(...)`](https://www.tensorflow.org/api_docs/python/tf/histogram_fixed_width_bins) | : Bins the given values for use in a histogram. | | |
| [`identity(...)`](https://www.tensorflow.org/api_docs/python/tf/identity) | : Return a Tensor with the same shape and contents as input. | | |
| [`identity_n(...)`](https://www.tensorflow.org/api_docs/python/tf/identity_n) | : Returns a list of tensors with the same shapes and contents as the input | | |
| [`import_graph_def(...)`](https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def) | : Imports the graph from `graph_def` into the current default `Graph`. (deprecated arguments) | | |
| [`init_scope(...)`](https://www.tensorflow.org/api_docs/python/tf/init_scope) | : A context manager that lifts ops out of control-flow scopes and function-building graphs. | | |
| [`inside_function(...)`](https://www.tensorflow.org/api_docs/python/tf/inside_function) | : Indicates whether the caller code is executing inside a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). | | |
| [`is_tensor(...)`](https://www.tensorflow.org/api_docs/python/tf/is_tensor) | : Checks whether `x` is a TF-native type that can be passed to many TF ops. | | |
| [`less(...)`](https://www.tensorflow.org/api_docs/python/tf/math/less) | : Returns the truth value of (x < y) element-wise. | | |
| [`less_equal(...)`](https://www.tensorflow.org/api_docs/python/tf/math/less_equal) | : Returns the truth value of (x <= y) element-wise. | | |
| [`linspace(...)`](https://www.tensorflow.org/api_docs/python/tf/linspace) | : Generates evenly-spaced values in an interval along a given axis. | | |
| [`load_library(...)`](https://www.tensorflow.org/api_docs/python/tf/load_library) | : Loads a TensorFlow plugin. | | |
| [`load_op_library(...)`](https://www.tensorflow.org/api_docs/python/tf/load_op_library) | : Loads a TensorFlow plugin, containing custom ops and kernels. | | |
| [`logical_and(...)`](https://www.tensorflow.org/api_docs/python/tf/math/logical_and) | : Logical AND function. | | |
| [`logical_not(...)`](https://www.tensorflow.org/api_docs/python/tf/math/logical_not) | : Returns the truth value of `NOT x` element-wise. | | |
| [`logical_or(...)`](https://www.tensorflow.org/api_docs/python/tf/math/logical_or) | : Returns the truth value of x OR y element-wise. | | |
| [`make_ndarray(...)`](https://www.tensorflow.org/api_docs/python/tf/make_ndarray) | : Create a numpy ndarray from a tensor. | | |
| [`make_tensor_proto(...)`](https://www.tensorflow.org/api_docs/python/tf/make_tensor_proto) | : Create a TensorProto. | | |
| [`map_fn(...)`](https://www.tensorflow.org/api_docs/python/tf/map_fn) | : Transforms `elems` by applying `fn` to each element unstacked on axis 0\. (deprecated arguments) | | |
| [`matmul(...)`](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul) | : Multiplies matrix `a` by matrix `b`, producing `a` * `b`. | | |
| [`matrix_square_root(...)`](https://www.tensorflow.org/api_docs/python/tf/linalg/sqrtm) | : Computes the matrix square root of one or more square matrices: | | |
| [`maximum(...)`](https://www.tensorflow.org/api_docs/python/tf/math/maximum) | : Returns the max of x and y (i.e. x > y ? x : y) element-wise. | | |
| [`meshgrid(...)`](https://www.tensorflow.org/api_docs/python/tf/meshgrid) | : Broadcasts parameters for evaluation on an N-D grid. | | |
| [`minimum(...)`](https://www.tensorflow.org/api_docs/python/tf/math/minimum) | : Returns the min of x and y (i.e. x < y ? x : y) element-wise. | | |
| [`multiply(...)`](https://www.tensorflow.org/api_docs/python/tf/math/multiply) | : Returns an element-wise x * y. | | |
| [`negative(...)`](https://www.tensorflow.org/api_docs/python/tf/math/negative) | : Computes numerical negative value element-wise. | | |
| [`no_gradient(...)`](https://www.tensorflow.org/api_docs/python/tf/no_gradient) | : Specifies that ops of type `op_type` is not differentiable. | | |
| [`no_op(...)`](https://www.tensorflow.org/api_docs/python/tf/no_op) | : Does nothing. Only useful as a placeholder for control edges. | | |
| [`nondifferentiable_batch_function(...)`](https://www.tensorflow.org/api_docs/python/tf/nondifferentiable_batch_function) | : Batches the computation done by the decorated function. | | |
| [`norm(...)`](https://www.tensorflow.org/api_docs/python/tf/norm) | : Computes the norm of vectors, matrices, and tensors. | | |
| [`not_equal(...)`](https://www.tensorflow.org/api_docs/python/tf/math/not_equal) | : Returns the truth value of (x != y) element-wise. | | |
| [`numpy_function(...)`](https://www.tensorflow.org/api_docs/python/tf/numpy_function) | : Wraps a python function and uses it as a TensorFlow op. | | |
| [`one_hot(...)`](https://www.tensorflow.org/api_docs/python/tf/one_hot) | : Returns a one-hot tensor. | | |
| [`ones(...)`](https://www.tensorflow.org/api_docs/python/tf/ones) | : Creates a tensor with all elements set to one (1). | | |
| [`ones_like(...)`](https://www.tensorflow.org/api_docs/python/tf/ones_like) | : Creates a tensor of all ones that has the same shape as the input. | | |
| [`pad(...)`](https://www.tensorflow.org/api_docs/python/tf/pad) | : Pads a tensor. | | |
| [`parallel_stack(...)`](https://www.tensorflow.org/api_docs/python/tf/parallel_stack) | : Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel. | | |
| [`pow(...)`](https://www.tensorflow.org/api_docs/python/tf/math/pow) | : Computes the power of one value to another. | | |
| [`print(...)`](https://www.tensorflow.org/api_docs/python/tf/print) | : Print the specified inputs. | | |
| [`py_function(...)`](https://www.tensorflow.org/api_docs/python/tf/py_function) | : Wraps a python function into a TensorFlow op that executes it eagerly. | | |
| [`quantize_and_dequantize_v4(...)`](https://www.tensorflow.org/api_docs/python/tf/quantize_and_dequantize_v4) | : Returns the gradient of `QuantizeAndDequantizeV4`. | | |
| [`range(...)`](https://www.tensorflow.org/api_docs/python/tf/range) | : Creates a sequence of numbers. | | |
| [`rank(...)`](https://www.tensorflow.org/api_docs/python/tf/rank) | : Returns the rank of a tensor. | | |
| [`realdiv(...)`](https://www.tensorflow.org/api_docs/python/tf/realdiv) | : Returns x / y element-wise for real types. | | |
| [`recompute_grad(...)`](https://www.tensorflow.org/api_docs/python/tf/recompute_grad) | : An eager-compatible version of recompute_grad. | | |
| [`reduce_all(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_all) | : Computes the "logical and" of elements across dimensions of a tensor. | | |
| [`reduce_any(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_any) | : Computes the "logical or" of elements across dimensions of a tensor. | | |
| [`reduce_logsumexp(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp) | : Computes log(sum(exp(elements across dimensions of a tensor))). | | |
| [`reduce_max(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max) | : Computes the maximum of elements across dimensions of a tensor. | | |
| [`reduce_mean(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean) | : Computes the mean of elements across dimensions of a tensor. | | |
| [`reduce_min(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_min) | : Computes the minimum of elements across dimensions of a tensor. | | |
| [`reduce_prod(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod) | : Computes the product of elements across dimensions of a tensor. | | |
| [`reduce_sum(...)`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum) | : Computes the sum of elements across dimensions of a tensor. | | |
| [`register_tensor_conversion_function(...)`](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function) | : Registers a function for converting objects of `base_type` to `Tensor`. | | |
| [`repeat(...)`](https://www.tensorflow.org/api_docs/python/tf/repeat) | : Repeat elements of `input`. | | |
| [`required_space_to_batch_paddings(...)`](https://www.tensorflow.org/api_docs/python/tf/required_space_to_batch_paddings) | : Calculate padding required to make block_shape divide input_shape. | | |
| [`reshape(...)`](https://www.tensorflow.org/api_docs/python/tf/reshape) | : Reshapes a tensor. | | |
| [`reverse(...)`](https://www.tensorflow.org/api_docs/python/tf/reverse) | : Reverses specific dimensions of a tensor. | | |
| [`reverse_sequence(...)`](https://www.tensorflow.org/api_docs/python/tf/reverse_sequence) | : Reverses variable length slices. | | |
| [`roll(...)`](https://www.tensorflow.org/api_docs/python/tf/roll) | : Rolls the elements of a tensor along an axis. | | |
| [`round(...)`](https://www.tensorflow.org/api_docs/python/tf/math/round) | : Rounds the values of a tensor to the nearest integer, element-wise. | | |
| [`saturate_cast(...)`](https://www.tensorflow.org/api_docs/python/tf/dtypes/saturate_cast) | : Performs a safe saturating cast of `value` to `dtype`. | | |
| [`scalar_mul(...)`](https://www.tensorflow.org/api_docs/python/tf/math/scalar_mul) | : Multiplies a scalar times a `Tensor` or `IndexedSlices` object. | | |
| [`scan(...)`](https://www.tensorflow.org/api_docs/python/tf/scan) | : scan on the list of tensors unpacked from `elems` on dimension 0\. (deprecated argument values) | | |
| [`scatter_nd(...)`](https://www.tensorflow.org/api_docs/python/tf/scatter_nd) | : Scatter `updates` into a new tensor according to `indices`. | | |
| [`searchsorted(...)`](https://www.tensorflow.org/api_docs/python/tf/searchsorted) | : Searches input tensor for values on the innermost dimension. | | |
| [`sequence_mask(...)`](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) | : Returns a mask tensor representing the first N positions of each cell. | | |
| [`shape(...)`](https://www.tensorflow.org/api_docs/python/tf/shape) | : Returns a tensor containing the shape of the input tensor. | | |
| [`shape_n(...)`](https://www.tensorflow.org/api_docs/python/tf/shape_n) | : Returns shape of tensors. | | |
| [`sigmoid(...)`](https://www.tensorflow.org/api_docs/python/tf/math/sigmoid) | : Computes sigmoid of `x` element-wise. | | |
| [`sign(...)`](https://www.tensorflow.org/api_docs/python/tf/math/sign) | : Returns an element-wise indication of the sign of a number. | | |
| [`sin(...)`](https://www.tensorflow.org/api_docs/python/tf/math/sin) | : Computes sine of x element-wise. | | |
| [`sinh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/sinh) | : Computes hyperbolic sine of x element-wise. | | |
| [`size(...)`](https://www.tensorflow.org/api_docs/python/tf/size) | : Returns the size of a tensor. | | |
| [`slice(...)`](https://www.tensorflow.org/api_docs/python/tf/slice) | : Extracts a slice from a tensor. | | |
| [`sort(...)`](https://www.tensorflow.org/api_docs/python/tf/sort) | : Sorts a tensor. | | |
| [`space_to_batch(...)`](https://www.tensorflow.org/api_docs/python/tf/space_to_batch) | : SpaceToBatch for N-D tensors of type T. | | |
| [`space_to_batch_nd(...)`](https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd) | : SpaceToBatch for N-D tensors of type T. | | |
| [`split(...)`](https://www.tensorflow.org/api_docs/python/tf/split) | : Splits a tensor `value` into a list of sub tensors. | | |
| [`sqrt(...)`](https://www.tensorflow.org/api_docs/python/tf/math/sqrt) | : Computes element-wise square root of the input tensor. | | |
| [`square(...)`](https://www.tensorflow.org/api_docs/python/tf/math/square) | : Computes square of x element-wise. | | |
| [`squeeze(...)`](https://www.tensorflow.org/api_docs/python/tf/squeeze) | : Removes dimensions of size 1 from the shape of a tensor. | | |
| [`stack(...)`](https://www.tensorflow.org/api_docs/python/tf/stack) | : Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor. | | |
| [`stop_gradient(...)`](https://www.tensorflow.org/api_docs/python/tf/stop_gradient) | : Stops gradient computation. | | |
| [`strided_slice(...)`](https://www.tensorflow.org/api_docs/python/tf/strided_slice) | : Extracts a strided slice of a tensor (generalized Python array indexing). | | |
| [`subtract(...)`](https://www.tensorflow.org/api_docs/python/tf/math/subtract) | : Returns x - y element-wise. | | |
| [`switch_case(...)`](https://www.tensorflow.org/api_docs/python/tf/switch_case) | : Create a switch/case operation, i.e. an integer-indexed conditional. | | |
| [`tan(...)`](https://www.tensorflow.org/api_docs/python/tf/math/tan) | : Computes tan of x element-wise. | | |
| [`tanh(...)`](https://www.tensorflow.org/api_docs/python/tf/math/tanh) | : Computes hyperbolic tangent of `x` element-wise. | | |
| [`tensor_scatter_nd_add(...)`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add) | : Adds sparse `updates` to an existing tensor according to `indices`. | | |
| [`tensor_scatter_nd_max(...)`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_max) | | | |
| [`tensor_scatter_nd_min(...)`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_min) | | | |
| [`tensor_scatter_nd_sub(...)`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_sub) | : Subtracts sparse `updates` from an existing tensor according to `indices`. | | |
| [`tensor_scatter_nd_update(...)`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) | : "Scatter `updates` into an existing tensor according to `indices`. | | |
| [`tensordot(...)`](https://www.tensorflow.org/api_docs/python/tf/tensordot) | : Tensor contraction of a and b along specified axes and outer product. | | |
| [`tile(...)`](https://www.tensorflow.org/api_docs/python/tf/tile) | : Constructs a tensor by tiling a given tensor. | | |
| [`timestamp(...)`](https://www.tensorflow.org/api_docs/python/tf/timestamp) | : Provides the time since epoch in seconds. | | |
| [`transpose(...)`](https://www.tensorflow.org/api_docs/python/tf/transpose) | : Transposes `a`, where `a` is a Tensor. | | |
| [`truediv(...)`](https://www.tensorflow.org/api_docs/python/tf/math/truediv) | : Divides x / y elementwise (using Python 3 division operator semantics). | | |
| [`truncatediv(...)`](https://www.tensorflow.org/api_docs/python/tf/truncatediv) | : Returns x / y element-wise for integer types. | | |
| [`truncatemod(...)`](https://www.tensorflow.org/api_docs/python/tf/truncatemod) | : Returns element-wise remainder of division. This emulates C semantics in that | | |
| [`tuple(...)`](https://www.tensorflow.org/api_docs/python/tf/tuple) | : Group tensors together. | | |
| [`type_spec_from_value(...)`](https://www.tensorflow.org/api_docs/python/tf/type_spec_from_value) | : Returns a [`tf.TypeSpec`](https://www.tensorflow.org/api_docs/python/tf/TypeSpec) that represents the given `value`. | | |
| [`unique(...)`](https://www.tensorflow.org/api_docs/python/tf/unique) | : Finds unique elements in a 1-D tensor. | | |
| [`unique_with_counts(...)`](https://www.tensorflow.org/api_docs/python/tf/unique_with_counts) | : Finds unique elements in a 1-D tensor. | | |
| [`unravel_index(...)`](https://www.tensorflow.org/api_docs/python/tf/unravel_index) | : Converts an array of flat indices into a tuple of coordinate arrays. | | |
| [`unstack(...)`](https://www.tensorflow.org/api_docs/python/tf/unstack) | : Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors. | | |
| [`variable_creator_scope(...)`](https://www.tensorflow.org/api_docs/python/tf/variable_creator_scope) | : Scope which defines a variable creation function to be used by variable(). | | |
| [`vectorized_map(...)`](https://www.tensorflow.org/api_docs/python/tf/vectorized_map) | : Parallel map on the list of tensors unpacked from `elems` on dimension 0. | | |
| [`where(...)`](https://www.tensorflow.org/api_docs/python/tf/where) | : Return the elements where `condition` is `True` (multiplexing `x` and `y`). | | |
| [`while_loop(...)`](https://www.tensorflow.org/api_docs/python/tf/while_loop) | : Repeat `body` while the condition `cond` is true. (deprecated argument values) | | |
| [`zeros(...)`](https://www.tensorflow.org/api_docs/python/tf/zeros) | : Creates a tensor with all elements set to zero. | | |
| [`zeros_like(...)`](https://www.tensorflow.org/api_docs/python/tf/zeros_like) | : Creates a tensor with all elements set to zero. |
