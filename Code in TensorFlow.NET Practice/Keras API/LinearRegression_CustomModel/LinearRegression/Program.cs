using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            TFNET tfnet = new TFNET();
            tfnet.Run();
        }
    }
    public class TFNET
    {
        public void Run()
        {
            Tensor X = tf.constant(new[,] { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
            Tensor y = tf.constant(new[,] { { 10.0f }, { 20.0f } });

            var model = new Linear(new ModelArgs());
            var optimizer = keras.optimizers.SGD(learning_rate: 0.01f);
            foreach (var step in range(20))//20 iterations for test
            {
                using var g = tf.GradientTape();
                var y_pred = model.Apply(X);//using "y_pred = model.Apply(X)" replace "y_pred = a * X + b"
                var loss = tf.reduce_mean(tf.square(y_pred - y));
                var grads = g.gradient(loss, model.trainable_variables);
                optimizer.apply_gradients(zip(grads, model.trainable_variables.Select(x => x as ResourceVariable)));
                print($"step: {step},loss: {loss.numpy()}");
            }
            print(model.trainable_variables.ToArray());
            Console.ReadKey();
        }
    }
    public class Linear : Model
    {
        Layer dense;
        public Linear(ModelArgs args) : base(args)
        {
            dense = keras.layers.Dense(1, activation: null,
                kernel_initializer: tf.zeros_initializer, bias_initializer: tf.zeros_initializer);
            StackLayers(dense);
        }
        // Set forward pass
        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            var outputs = dense.Apply(inputs);
            return outputs;
        }
    }
}
