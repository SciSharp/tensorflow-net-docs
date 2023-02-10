open NumSharp
open type Tensorflow.Binding

// A very simple "hello world" using TensorFlow v2 tensors.
let private run () =
    // Eager model is enabled by default.
    //tf.enable_eager_execution()

    (* Create a Constant op
        The op is added as a node to the default graph.

        The value returned by the constructor represents the output
        of the Constant op. *)
    let str = "Hello, TensorFlow.NET!"
    let hello = tf.constant(str)

    // tf.Tensor: shape=(), dtype=string, numpy=b'Hello, TensorFlow.NET!'
    //print(hello);

    let hello_str = NDArray.AsStringArray(hello.numpy()).[0]

    print(hello_str)

run()