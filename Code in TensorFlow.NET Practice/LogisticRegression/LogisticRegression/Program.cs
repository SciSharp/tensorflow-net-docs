using System;
using Tensorflow;
using static Tensorflow.Binding;


namespace LogisticRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            int training_epochs = 1000;
            int? train_size = null;
            int validation_size = 5000;
            int? test_size = null;
            int batch_size = 256;
            int num_classes = 10; // 0 to 9 digits
            int num_features = 784; // 28*28
            float learning_rate = 0.01f;
            int display_step = 50;
            float accuracy = 0f;

            Datasets<MnistDataSet> mnist;

            // Prepare MNIST data.From http://yann.lecun.com/exdb/mnist/
            var ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();
            // Flatten images to 1-D vector of 784 features (28*28).
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            // Use tf.data API to shuffle and batch data.
            var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1);

            // Weight of shape [784, 10], the 28*28 image features, and total number of classes.
            var W = tf.Variable(tf.ones((num_features, num_classes)), name: "weight");
            // Bias of shape [10], the total number of classes.
            var b = tf.Variable(tf.zeros(num_classes), name: "bias");

            Func<Tensor, Tensor> logistic_regression = x
            => tf.nn.softmax(tf.matmul(x, W) + b);

            Func<Tensor, Tensor, Tensor> cross_entropy = (y_pred, y_true) =>
            {
                y_true = tf.cast(y_true, TF_DataType.TF_UINT8);
                // Encode label to a one hot vector.
                y_true = tf.one_hot(y_true, depth: num_classes);
                // Clip prediction values to avoid log(0) error.
                y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
                // Compute cross-entropy.
                return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1));
            };

            Func<Tensor, Tensor, Tensor> Accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
            };

            // Stochastic gradient descent optimizer.
            var optimizer = tf.optimizers.SGD(learning_rate);

            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                var pred = logistic_regression(x);
                var loss = cross_entropy(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, (W, b));

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, (W, b)));
            };

            train_data = train_data.take(training_epochs);
            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = logistic_regression(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = Accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                    accuracy = acc.numpy();
                }
            }

            // Test model on validation set.
            {
                var pred = logistic_regression(x_test);
                print($"Test Accuracy: {(float)Accuracy(pred, y_test)}");
            }
        }
    }
}
