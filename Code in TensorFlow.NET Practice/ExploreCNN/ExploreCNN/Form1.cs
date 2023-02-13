using NumSharp;
using System;
using System.Windows.Forms;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using Tensorflow;

namespace ExploreCNN
{
    public partial class Form1 : Form
    {
        Model model;
        LayersApi layers = new LayersApi();
        NDArray x_train, y_train, x_test, y_test, x_test_raw;//x_test_raw for image show
        const string modelFile = "model.wts";

        public Form1()
        {
            InitializeComponent();
        }

        private void button_loaddata_Click(object sender, EventArgs e)
        {
            this.button_loaddata.Text = "loading...";
            this.Enabled = false;
            this.Cursor = Cursors.WaitCursor;
            // Step-1. Prepare Data
            (x_train, y_train, x_test_raw, y_test) = keras.datasets.mnist.load_data();
            x_train = x_train.reshape(60000, 28, 28, 1) / 255f;
            x_test = x_test_raw.reshape(10000, 28, 28, 1) / 255f;
            this.button_loaddata.Text = "Load Data";
            this.Enabled = true;
            this.Cursor = Cursors.Default;
        }

        private void button_train_Click(object sender, EventArgs e)
        {
            this.button_train.Text = "training...";
            this.Enabled = false;
            this.Cursor = Cursors.WaitCursor;
            var outputText = new System.Text.StringBuilder();
            tf_output_redirect = new System.IO.StringWriter(outputText);
            bool isStoped = false;

            System.Threading.Tasks.Task.Run(() =>
            {
                model = CreateModel();
                // train model by feeding data and labels.
                model.fit(x_train, y_train, batch_size: 64, epochs: 1, validation_split: 0.2f);
                // evluate the model
                model.evaluate(x_test, y_test, verbose: 2);
                System.Threading.Thread.Sleep(1000);
                isStoped = true;
            });

            System.Threading.Tasks.Task.Run(() =>
            {
                var preLength = outputText.Length;
                while (!isStoped)
                {
                    System.Threading.Thread.Sleep(100);
                    var curLength = outputText.Length;
                    if (preLength < curLength)
                    {
                        this.Invoke(new Action(() =>
                        {
                            textBox_history.Text = outputText.ToString();
                            TextBox_Top();
                            preLength = curLength;
                        }));
                    }
                }

                tf_output_redirect.Close();
                tf_output_redirect.Dispose();
                tf_output_redirect = null;//dispose

                model.save_weights(modelFile, true);

                this.Invoke(new Action(() =>
                {
                    this.button_train.Text = "Train";
                    this.Enabled = true;
                    this.Cursor = Cursors.Default;
                }));
            });

        }

        /// <summary>
        /// Step-2. Build CNN Model with Keras Functional
        /// </summary>
        /// <returns></returns>
        private Model CreateModel()
        {
            // input layer
            var inputs = keras.Input(shape: (28, 28, 1));
            // 1st convolution layer
            var outputs = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu).Apply(inputs);
            // 2nd maxpooling layer
            outputs = layers.MaxPooling2D(2, strides: 2).Apply(outputs);
            // 3nd convolution layer
            outputs = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu).Apply(outputs);
            // 4nd maxpooling layer
            outputs = layers.MaxPooling2D(2, strides: 2).Apply(outputs);
            // 5nd flatten layer
            outputs = layers.Flatten().Apply(outputs);
            // 6nd dense layer
            outputs = layers.Dense(128).Apply(outputs);
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

            return model;
        }

        private void button_showcnn_Click(object sender, EventArgs e)
        {
            int num = Convert.ToInt32(numericUpDown_image.Value);
            int conv_num = Convert.ToInt32(numericUpDown_CONVOLUTION_NUMBER.Value);

            //show raw image
            var bmp = GrayToRGB(x_test_raw[num]).ToBitmap();
            pictureBox_Image.Image = bmp;

            //clear Graph
            tf.Context.reset_context();
            if (System.IO.File.Exists(modelFile))
            {
                model = CreateModel();
                model.load_weights(modelFile);
            }

            //show predict result
            textBox_history.Text += "\r\n" + "Real Label is：" + y_test[num] + "\r\n";
            var predict_result = model.predict(x_test[num].reshape(new[] { 1, 28, 28, 1 }));
            var predict_label = np.argmax(predict_result[0].numpy(), axis: 1);
            textBox_history.Text += "\r\n" + "Predict Label is：" + predict_label.ToString() + "\r\n";
            TextBox_Top();

            //something interesting : show cnn processing image
            Tensor[] layer_outputs = new Tensor[4];
            Tensor layer_inputs = ((Layer)model.Layers[1]).input[0];
            for (int i = 0; i < 4; i++)
                layer_outputs[i] = ((Layer)model.Layers[i + 2]).input[0];
            var activation_model = keras.Model(inputs: layer_inputs, outputs: layer_outputs);

            //show layer of convolution 1#
            var c1 = activation_model.predict(x_test[num].reshape(new[] { 1, 28, 28, 1 }))[0];
            var np_c1 = np.squeeze(Clip(c1.numpy()["0", ":", ":", conv_num.ToString()] * 255));
            var bmp_c1 = GrayToRGB(np_c1).ToBitmap();
            pictureBox_conv1.Image = bmp_c1;
            label_conv1.Text = bmp_c1.Width.ToString() + " * " + bmp_c1.Height.ToString();

            //show layer of maxpooling 1#
            var p1 = activation_model.predict(x_test[num].reshape(new[] { 1, 28, 28, 1 }))[1];
            var np_p1 = np.squeeze(Clip(p1.numpy()["0", ":", ":", conv_num.ToString()] * 255));
            var bmp_p1 = GrayToRGB(np_p1).ToBitmap(); ;
            pictureBox_pooling1.Image = bmp_p1;
            label_pooling1.Text = bmp_p1.Width.ToString() + " * " + bmp_p1.Height.ToString();

            //show layer of convolution 2#
            var c2 = activation_model.predict(x_test[num].reshape(new[] { 1, 28, 28, 1 }))[2];
            var np_c2 = np.squeeze(Clip(c2.numpy()["0", ":", ":", conv_num.ToString()] * 255));
            var bmp_c2 = GrayToRGB(np_c2).ToBitmap();
            pictureBox_conv2.Image = bmp_c2;
            label_conv2.Text = bmp_c2.Width.ToString() + " * " + bmp_c2.Height.ToString();

            //show layer of maxpooling 2#
            var p2 = activation_model.predict(x_test[num].reshape(new[] { 1, 28, 28, 1 }))[3];
            var np_p2 = np.squeeze(Clip(p2.numpy()["0", ":", ":", conv_num.ToString()] * 255));
            var bmp_p2 = GrayToRGB(np_p2).ToBitmap();
            pictureBox_pooling2.Image = bmp_p2;
            label_pooling2.Text = bmp_p2.Width.ToString() + " * " + bmp_p2.Height.ToString();
        }

        private NDArray GrayToRGB(NDArray img2D)
        {
            var img4A = np.full_like(img2D, (byte)255);
            var img3D = np.expand_dims(img2D, 2);
            var r = np.dstack(img3D, img3D, img3D, img4A);
            var img4 = np.expand_dims(r, 0);
            return img4;
        }
        private NDArray Clip(NDArray nd_input)
        {
            var nd_min = np.full_like(nd_input, (byte)0);
            var nd_max = np.full_like(nd_input, (byte)255);
            nd_input = np.clip(nd_input, nd_min, nd_max);
            return nd_input.astype(NPTypeCode.Byte);
        }

        private void TextBox_Top()
        {
            textBox_history.SelectionStart = textBox_history.Text.Length;
            textBox_history.SelectionLength = 0;
            textBox_history.ScrollToCaret();
        }
    }
}
