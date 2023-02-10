open NumSharp
open Tensorflow
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Layers
open type Tensorflow.KerasApi

type Tensor with
    member x.asTensors : Tensors = new Tensors([| x |])

let prepareData () =
    let (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data().Deconstruct()
    let x_train = x_train.reshape(60000, 784) / 255f
    let x_test = x_test.reshape(10000, 784) / 255f
    (x_train, y_train, x_test, y_test)

let buildModel () =
    // input layer
    let inputs = keras.Input(shape = TensorShape 784)

    let layers = LayersApi()

    // 1st dense layer
    let outputs = layers.Dense(64, activation = keras.activations.Relu).Apply(inputs.asTensors)

    // 2nd dense layer
    let outputs = layers.Dense(64, activation = keras.activations.Relu).Apply(outputs)

    // output layer
    let outputs = layers.Dense(10).Apply(outputs)

    // build keras model
    let model = keras.Model(inputs.asTensors, outputs, name = "mnist_model")
    // show model summary
    model.summary()

    // compile keras model into tensorflow's static graph
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = true),
        optimizer = keras.optimizers.RMSprop(),
        metrics = [| "accuracy" |])

    model

let private train (x_train : NDArray, y_train) (x_test : NDArray, y_test) (model : Functional) =
    // train model by feeding data and labels.
    model.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_split = 0.2f)

    // evluate the model
    model.evaluate(x_test, y_test, verbose = 2)

    // save and serialize model
    model.save("mnist_model")

    // recreate the exact same model purely from the file:
    // model = keras.models.load_model("mnist_model")

let private run () =
    let (x_train, y_train, x_test, y_test) = prepareData()
    let model = buildModel()
    train (x_train, y_train) (x_test, y_test) model

run()