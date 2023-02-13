using NumSharp;
using System;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Layers;

namespace DNN_Keras_Functional
{
    class Program
    {
        static void Main(string[] args)
        {
            DNN_Keras_Functional dnn = new DNN_Keras_Functional();
            dnn.Main();
        }
        class DNN_Keras_Functional
        {
            Model model;
            NDArray x_train, y_train, x_test, y_test;
            LayersApi layers = new LayersApi();
            public void Main()
            {
                //1. prepare data
                (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
                x_train = x_train.reshape(60000, 784) / 255f;
                x_test = x_test.reshape(10000, 784) / 255f;

                //2. buid model                
                var inputs = keras.Input(shape: 784);// input layer 
                var outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);// 1st dense layer 
                outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);// 2nd dense layer 
                outputs = layers.Dense(10).Apply(outputs);// output layer
                model = keras.Model(inputs, outputs, name: "mnist_model");// build keras model   
                model.summary();// show model summary
                model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                    optimizer: keras.optimizers.RMSprop(),
                    metrics: new[] { "accuracy" });// compile keras model into tensorflow's static graph

                //3. train model by feeding data and labels.
                model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);

                //4. evluate the model
                model.evaluate(x_test, y_test, verbose: 2);

                //5. save and serialize model
                model.save("mnist_model");

                // reload the exact same model purely from the file:
                // model = keras.models.load_model("path_to_my_model");

                Console.ReadKey();
            }
        }
    }
}
