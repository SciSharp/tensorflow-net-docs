using NumSharp;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using static Tensorflow.KerasApi;

namespace MnistCNNKerasFunctional
{
    class Program
    {
        static void Main(string[] args)
        {
            MnistCNN cnn = new MnistCNN();
            cnn.Main();
        }

        class MnistCNN
        {
            Model model;
            LayersApi layers = new LayersApi();
            NDArray x_train, y_train, x_test, y_test;
            public void Main()
            {
                // Step-1. Prepare Data
                (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
                x_train = x_train.reshape(60000, 28, 28, 1) / 255f;
                x_test = x_test.reshape(10000, 28, 28, 1) / 255f;

                // Step-2. Build CNN Model with Keras Functional
                // input layer
                var inputs = keras.Input(shape: (28, 28, 1));
                // 1st convolution layer
                var outputs = layers.Conv2D(32, kernel_size: 5, activation: keras.activations.Relu).Apply(inputs);
                // 2nd maxpooling layer
                outputs = layers.MaxPooling2D(2, strides: 2).Apply(outputs);
                // 3nd convolution layer
                outputs = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu).Apply(outputs);
                // 4nd maxpooling layer
                outputs = layers.MaxPooling2D(2, strides: 2).Apply(outputs);
                // 5nd flatten layer
                outputs = layers.Flatten().Apply(outputs);
                // 6nd dense layer
                outputs = layers.Dense(1024).Apply(outputs);
                // 7nd dropout layer
                outputs = layers.Dropout(rate: 0.5f).Apply(outputs);
                // output layer
                outputs = layers.Dense(10).Apply(outputs);
                // build keras model
                model = keras.Model(inputs, outputs, name: "mnist_model");
                // show model summary
                model.summary();
                // compile keras model into tensorflow's static graph
                model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                    optimizer: keras.optimizers.Adam(learning_rate: 0.001f),
                    metrics: new[] { "accuracy" });

                // Step-3. Train Model
                // train model by feeding data and labels.
                model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);
                // evluate the model
                model.evaluate(x_test, y_test, verbose: 2);

            }
        }
    }
}
