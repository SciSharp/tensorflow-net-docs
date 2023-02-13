using NumSharp;
using System;
using static Tensorflow.Binding;

namespace LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            //1. Prepare data
            NDArray train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                         7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
            NDArray train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                         2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
            int n_samples = train_X.shape[0];

            //2. Set weights
            var W = tf.Variable(0f, name: "weight");
            var b = tf.Variable(0f, name: "bias");
            float learning_rate = 0.01f;
            var optimizer = tf.optimizers.SGD(learning_rate);

            //3. Run the optimization to update weights
            int training_steps = 1000;
            int display_step = 50;
            foreach (var step in range(1, training_steps + 1))
            {
                using var g = tf.GradientTape();
                // Linear regression (Wx + b).
                var pred = W * train_X + b;
                // MSE:Mean square error.
                var loss = tf.reduce_sum(tf.pow(pred - train_Y, 2)) / n_samples;
                var gradients = g.gradient(loss, (W, b));

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, (W, b)));
                if (step % display_step == 0)
                {
                    pred = W * train_X + b;
                    loss = tf.reduce_sum(tf.pow(pred - train_Y, 2)) / n_samples;
                    print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
                }
            }


            Console.ReadKey();


        }
    }
}
