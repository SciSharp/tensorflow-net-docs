using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;

namespace DNN_Eager
{
    class Program
    {
        static void Main(string[] args)
        {
            DNN_Eager dnn = new DNN_Eager();
            dnn.Main();
        }
    }

    class DNN_Eager
    {
        int num_classes = 10; // MNIST 的字符类别 0~9 总共 10 类
        int num_features = 784; // 输入图像的特征尺寸，即像素28*28=784

        // 超参数
        float learning_rate = 0.001f;// 学习率
        int training_steps = 1000;// 训练轮数
        int batch_size = 256;// 批次大小
        int display_step = 100;// 训练数据 显示周期

        // 神经网络参数
        int n_hidden_1 = 128; // 第1层隐藏层的神经元数量
        int n_hidden_2 = 256; // 第2层隐藏层的神经元数量

        IDatasetV2 train_data;// MNIST 数据集
        NDArray x_test, y_test, x_train, y_train;// 数据集拆分为训练集和测试集
        IVariableV1 h1, h2, wout, b1, b2, bout;// 待训练的权重变量
        float accuracy_test = 0f;// 测试集准确率

        public void Main()
        {
            ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();// 下载 或 加载本地 MNIST
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));// 输入数据展平
            (x_train, x_test) = (x_train / 255f, x_test / 255f);// 归一化

            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);//转换为 Dataset 格式
            train_data = train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps);// 数据预处理

            // 随机初始化网络权重变量，并打包成数组方便后续梯度求导作为参数。
            var random_normal = tf.initializers.random_normal_initializer();
            h1 = tf.Variable(random_normal.Apply(new InitializerArgs((num_features, n_hidden_1))));
            h2 = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_1, n_hidden_2))));
            wout = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_2, num_classes))));
            b1 = tf.Variable(tf.zeros(n_hidden_1));
            b2 = tf.Variable(tf.zeros(n_hidden_2));
            bout = tf.Variable(tf.zeros(num_classes));
            var trainable_variables = new IVariableV1[] { h1, h2, wout, b1, b2, bout };

            // 采用随机梯度下降优化器
            var optimizer = tf.optimizers.SGD(learning_rate);

            // 训练模型
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // 运行优化器 进行模型权重 w 和 b 的更新
                run_optimization(optimizer, batch_x, batch_y, trainable_variables);

                if (step % display_step == 0)
                {
                    var pred = neural_net(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // 在测试集上对训练后的模型进行预测准确率性能评估
            {
                var pred = neural_net(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

        }

        // 运行优化器
        void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y, IVariableV1[] trainable_variables)
        {
            using var g = tf.GradientTape();
            var pred = neural_net(x);
            var loss = cross_entropy(pred, y);

            // 计算梯度
            var gradients = g.gradient(loss, trainable_variables);

            // 更新模型权重 w 和 b 
            var a = zip(gradients, trainable_variables.Select(x => x as ResourceVariable));
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
        }

        // 模型预测准确度
        Tensor accuracy(Tensor y_pred, Tensor y_true)
        {
            // 使用 argmax 提取预测概率最大的标签，和实际值比较，计算模型预测的准确度
            var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        // 搭建网络模型
        Tensor neural_net(Tensor x)
        {
            // 第1层隐藏层采用128个神经元。
            var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
            // 使用 sigmoid 激活函数，增加层输出的非线性特征
            layer_1 = tf.nn.sigmoid(layer_1);

            // 第2层隐藏层采用256个神经元。
            var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
            // 使用 sigmoid 激活函数，增加层输出的非线性特征
            layer_2 = tf.nn.sigmoid(layer_2);

            // 输出层的神经元数量和标签类型数量相同
            var out_layer = tf.matmul(layer_2, wout.AsTensor()) + bout.AsTensor();
            // 使用 Softmax 函数将输出类别转换为各类别的概率分布
            return tf.nn.softmax(out_layer);
        }

        // 交叉熵损失函数
        Tensor cross_entropy(Tensor y_pred, Tensor y_true)
        {
            // 将标签转换为One-Hot格式
            y_true = tf.one_hot(y_true, depth: num_classes);
            // 保持预测值在 1e-9 和 1.0 之间，防止值下溢出现log(0)报错
            y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
            // 计算交叉熵损失
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)));
        }

    }
}
