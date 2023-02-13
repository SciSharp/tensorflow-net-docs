# Introduction

## Why TensorFlow in C# and F# ?

`SciSharp STACK`'s mission is to bring popular data science technology into the .NET world and to provide .NET developers with a powerful Machine Learning tool set without reinventing the wheel. Since the APIs are kept as similar as possible you can immediately adapt any existing Tensorflow code in C# or F# with a zero learning curve. Take a look at a comparison picture and see how comfortably a Tensorflow/Python script translates into a C# program with TensorFlow.NET.

> ![pythn vs csharp](../_media/syntax-comparision.png)

SciSharp's philosophy allows a large number of machine learning code written in Python to be quickly migrated to .NET, enabling .NET developers to use cutting edge machine learning models and access a vast number of Tensorflow resources which would not be possible without this project.

> <video style="width: 100%;" src="_media/csharp-vs-python-speed.mp4" type="video/mp4" controls autoplay loop>python vs csharp on speed</video>
>
> No surprise, .NET version (left window) is faster than python version (right window) binding. 10K iterates for linear regression in TensorFlow SGD. (CPU)

> ![python vs csharp on speed and memory](../_media/csharp_vs_python_speed_memory.jpg)
>
> It is 2x faster and 1/4 memory occupation of training time in eager mode than python binding. (TensorFlow.NET 0.20-preview2)

## Why over TensorFlowSharp?

In comparison to other projects, like for instance [TensorFlowSharp](https://www.nuget.org/packages/TensorFlowSharp/) which only provide Tensorflow's low-level C++ API and can only run models that were built using Python, Tensorflow.NET also implements Tensorflow's high level API where all the magic happens. This computation graph building layer is still under active development. Once it is completely implemented you can build new Machine Learning models in C# or F#. 

## Getting Started

> Make sure all the dependencies are properly installed! [Installation](essentials/installation.md)

### C# Example

Import TF.NET and Keras API in your project.

```csharp
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
```

Linear Regression in `Eager` mode:

```csharp
// Parameters        
var training_steps = 1000;
var learning_rate = 0.01f;
var display_step = 100;

// Sample data
var X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
             7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
var Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
             2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
var n_samples = X.shape[0];

// We can set a fixed init value in order to demo
var W = tf.Variable(-0.06f, name: "weight");
var b = tf.Variable(-0.73f, name: "bias");
var optimizer = keras.optimizers.SGD(learning_rate);

// Run training for the given number of steps.
foreach (var step in range(1, training_steps + 1))
{
    // Run the optimization to update W and b values.
    // Wrap computation inside a GradientTape for automatic differentiation.
    using var g = tf.GradientTape();
    // Linear regression (Wx + b).
    var pred = W * X + b;
    // Mean square error.
    var loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
    // should stop recording
    // Compute gradients.
    var gradients = g.gradient(loss, (W, b));

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, (W, b)));

    if (step % display_step == 0)
    {
        pred = W * X + b;
        loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
        print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
    }
}
```

Run this example in [Jupyter Notebook](https://github.com/SciSharp/SciSharpCube).

Toy version of `ResNet` in `Keras` functional API:

```csharp
// input layer
var inputs = keras.Input(shape: (32, 32, 3), name: "img");

// convolutional layer
var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
x = layers.Conv2D(64, 3, activation: "relu").Apply(x);
var block_1_output = layers.MaxPooling2D(3).Apply(x);

x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
var block_2_output = layers.add(x, block_1_output);

x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
var block_3_output = layers.add(x, block_2_output);

x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
x = layers.GlobalAveragePooling2D().Apply(x);
x = layers.Dense(256, activation: "relu").Apply(x);
x = layers.Dropout(0.5f).Apply(x);

// output layer
var outputs = layers.Dense(10).Apply(x);

// build keras model
model = keras.Model(inputs, outputs, name: "toy_resnet");
model.summary();

// compile keras model in tensorflow static graph
model.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
	loss: keras.losses.CategoricalCrossentropy(from_logits: true),
	metrics: new[] { "acc" });

// prepare dataset
var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();

// training
model.fit(x_train[new Slice(0, 1000)], y_train[new Slice(0, 1000)], 
          batch_size: 64, 
          epochs: 10, 
          validation_split: 0.2f);
```

### F# Example

Linear Regression in `Eager` mode:

```fsharp
#r "nuget: TensorFlow.Net"
#r "nuget: TensorFlow.Keras"
#r "nuget: SciSharp.TensorFlow.Redist"
#r "nuget: NumSharp"

open NumSharp
open Tensorflow
open type Tensorflow.Binding
open type Tensorflow.KerasApi

let tf = New<tensorflow>()
tf.enable_eager_execution()

// Parameters
let training_steps = 1000
let learning_rate = 0.01f
let display_step = 100

// Sample data
let train_X = 
    np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
             7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f)
let train_Y = 
    np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
             2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f)
let n_samples = train_X.shape.[0]

// We can set a fixed init value in order to demo
let W = tf.Variable(-0.06f,name = "weight")
let b = tf.Variable(-0.73f, name = "bias")
let optimizer = keras.optimizers.SGD(learning_rate)

// Run training for the given number of steps.
for step = 1 to  (training_steps + 1) do 
    // Run the optimization to update W and b values.
    // Wrap computation inside a GradientTape for automatic differentiation.
    use g = tf.GradientTape()
    // Linear regression (Wx + b).
    let pred = W * train_X + b
    // Mean square error.
    let loss = tf.reduce_sum(tf.pow(pred - train_Y,2)) / (2 * n_samples)
    // should stop recording
    // compute gradients
    let gradients = g.gradient(loss,struct (W,b))

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, struct (W,b)))

    if (step % display_step) = 0 then
        let pred = W * train_X + b
        let loss = tf.reduce_sum(tf.pow(pred-train_Y,2)) / (2 * n_samples)
        printfn $"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}"
```

## External Documentations

Want to learn more? Read the docs & book [The Definitive Guide to Tensorflow.NET](https://tensorflownet.readthedocs.io/en/latest/FrontCover.html).
Want to try the Chinese version? Try [C# TensorFlow 2 入门教程](https://github.com/SciSharp/TensorFlow.NET-Tutorials).

## Complete Examples

Talk is cheap, [show me the code](https://github.com/SciSharp/SciSharp-Stack-Examples). (Repository of complete samples)
