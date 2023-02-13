using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.Sessions;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace CnnTextClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            CnnTextClassification word_cnn = new CnnTextClassification();
            word_cnn.Run();
        }
    }
    public class CnnTextClassification
    {
        public int? DataLimit = null;

        const string dataDir = "cnn_text";
        string TRAIN_PATH = $"{dataDir}/dbpedia_csv/train.csv";
        string TEST_PATH = $"{dataDir}/dbpedia_csv/test.csv";

        int NUM_CLASS = 14;
        int BATCH_SIZE = 64;
        int NUM_EPOCHS = 10;
        int WORD_MAX_LEN = 100;

        float loss_value = 0;
        double max_accuracy = 0;

        int vocabulary_size = -1;
        NDArray train_x, test_x, train_y, test_y;
        Dictionary<string, int> word_dict;

        public bool Run()
        {
            tf.compat.v1.disable_eager_execution();

            PrepareData();
            Train();
            Test();
            Predict();

            return max_accuracy > 0.9;
        }
        private void PrepareData()
        {
            // full dataset https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/dbpedia_subset.zip";
            Web.Download(url, dataDir, "dbpedia_subset.zip");
            Compress.UnZip(Path.Combine(dataDir, "dbpedia_subset.zip"), Path.Combine(dataDir, "dbpedia_csv"));

            Console.WriteLine("Building dataset...");
            var (x, y) = (new int[0][], new int[0]);

            word_dict = build_word_dict(TRAIN_PATH);
            vocabulary_size = len(word_dict);
            (x, y) = build_word_dataset(TRAIN_PATH, word_dict, WORD_MAX_LEN);

            Console.WriteLine("\tDONE ");

            (train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size: 0.15f);
            Console.WriteLine("Training set size: " + train_x.shape[0]);
            Console.WriteLine("Test set size: " + test_x.shape[0]);
        }
        private Dictionary<string, int> build_word_dict(string path)
        {
            var contents = File.ReadAllLines(path);

            var words = new List<string>();
            foreach (var content in contents)
                words.AddRange(clean_str(content).Split(' ').Where(x => x.Length > 1));
            var word_counter = words.GroupBy(x => x)
                .Select(x => new { Word = x.Key, Count = x.Count() })
                .OrderByDescending(x => x.Count)
                .ToArray();

            var word_dict = new Dictionary<string, int>();
            word_dict["<pad>"] = 0;
            word_dict["<unk>"] = 1;
            word_dict["<eos>"] = 2;
            foreach (var word in word_counter)
                word_dict[word.Word] = word_dict.Count;

            return word_dict;
        }
        private string clean_str(string str)
        {
            str = Regex.Replace(str, "[^A-Za-z0-9(),!?]", " ");
            str = Regex.Replace(str, ",", " ");
            return str;
        }
        private (int[][], int[]) build_word_dataset(string path, Dictionary<string, int> word_dict, int document_max_len)
        {
            var contents = File.ReadAllLines(path);
            var x = contents.Select(c => (clean_str(c) + " <eos>")
                .Split(' ').Take(document_max_len)
                .Select(w => word_dict.ContainsKey(w) ? word_dict[w] : word_dict["<unk>"]).ToArray())
                .ToArray();

            for (int i = 0; i < x.Length; i++)
                if (x[i].Length == document_max_len)
                    x[i][document_max_len - 1] = word_dict["<eos>"];
                else
                    Array.Resize(ref x[i], document_max_len);

            var y = contents.Select(c => int.Parse(c.Substring(0, c.IndexOf(','))) - 1).ToArray();

            return (x, y);
        }
        private (NDArray, NDArray, NDArray, NDArray) train_test_split(NDArray x, NDArray y, float test_size = 0.3f)
        {
            Console.WriteLine("Splitting in Training and Testing data...");
            int len = x.shape[0];
            int train_size = (int)Math.Round(len * (1 - test_size));
            train_x = x[new Slice(stop: train_size), new Slice()];
            test_x = x[new Slice(start: train_size), new Slice()];
            train_y = y[new Slice(stop: train_size)];
            test_y = y[new Slice(start: train_size)];
            Console.WriteLine("\tDONE");

            return (train_x, test_x, train_y, test_y);
        }
        private void Train()
        {
            var graph = BuildGraph();

            using (var sess = tf.Session(graph))
            {
                sess.run(tf.global_variables_initializer());
                var saver = tf.train.Saver(tf.global_variables());

                var train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS);
                var num_batches_per_epoch = (len(train_x) - 1) / BATCH_SIZE + 1;

                Tensor is_training = graph.OperationByName("is_training");
                Tensor model_x = graph.OperationByName("x");
                Tensor model_y = graph.OperationByName("y");
                Tensor loss = graph.OperationByName("loss/Mean");
                Operation optimizer = graph.OperationByName("loss/Adam");
                Tensor global_step = graph.OperationByName("Variable");
                Tensor accuracy = graph.OperationByName("accuracy/accuracy");

                var sw = new Stopwatch();
                sw.Start();

                int step = 0;
                foreach (var (x_batch, y_batch, total) in train_batches)
                {
                    (_, step, loss_value) = sess.run((optimizer, global_step, loss),
                        (model_x, x_batch), (model_y, y_batch), (is_training, true));
                    if (step % 10 == 0)
                    {
                        Console.WriteLine($"Training on batch {step}/{total} loss: {loss_value.ToString("0.0000")} {sw.ElapsedMilliseconds}ms.");
                        sw.Restart();
                    }

                    if (step % 100 == 0)
                    {
                        // Test accuracy with validation data for each epoch.
                        var valid_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1);
                        var (sum_accuracy, cnt) = (0.0f, 0);
                        foreach (var (valid_x_batch, valid_y_batch, total_validation_batches) in valid_batches)
                        {
                            var valid_feed_dict = new FeedDict
                            {
                                [model_x] = valid_x_batch,
                                [model_y] = valid_y_batch,
                                [is_training] = false
                            };
                            float accuracy_value = sess.run(accuracy, (model_x, valid_x_batch), (model_y, valid_y_batch), (is_training, false));
                            sum_accuracy += accuracy_value;
                            cnt += 1;
                        }

                        var valid_accuracy = sum_accuracy / cnt;

                        print($"\nValidation Accuracy = {valid_accuracy.ToString("P")}\n");

                        // Save model
                        if (valid_accuracy > max_accuracy)
                        {
                            max_accuracy = valid_accuracy;
                            saver.save(sess, $"{dataDir}/word_cnn.ckpt", global_step: step);
                            print("Model is saved.\n");
                        }
                    }
                }
            }
        }
        private Graph BuildGraph()
        {
            var graph = tf.Graph().as_default();

            WordCnn(vocabulary_size, WORD_MAX_LEN, NUM_CLASS);

            return graph;
        }
        private void WordCnn(int vocabulary_size, int document_max_len, int num_class)
        {
            var embedding_size = 128;
            var learning_rate = 0.001f;
            var filter_sizes = new int[3, 4, 5];
            var num_filters = 100;

            var x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            var y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            var is_training = tf.placeholder(tf.@bool, new TensorShape(), name: "is_training");
            var global_step = tf.Variable(0, trainable: false);
            var keep_prob = tf.where(is_training, 0.5f, 1.0f);
            Tensor x_emb = null;

            tf_with(tf.name_scope("embedding"), scope =>
            {
                var init_embeddings = tf.random_uniform(new int[] { vocabulary_size, embedding_size });
                var embeddings = tf.compat.v1.get_variable("embeddings", initializer: init_embeddings);
                x_emb = tf.nn.embedding_lookup(embeddings, x);
                x_emb = tf.expand_dims(x_emb, -1);
            });

            var pooled_outputs = new List<Tensor>();
            for (int len = 0; len < filter_sizes.Rank; len++)
            {
                int filter_size = filter_sizes.GetLength(len);
                var conv = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new int[] { filter_size, embedding_size },
                    strides: new int[] { 1, 1 },
                    padding: "VALID",
                    activation: tf.nn.relu).Apply(x_emb);

                var pool = keras.layers.max_pooling2d(
                    conv,
                    pool_size: new[] { document_max_len - filter_size + 1, 1 },
                    strides: new[] { 1, 1 },
                    padding: "VALID");

                pooled_outputs.Add(pool);
            }

            var h_pool = tf.concat(pooled_outputs, 3);
            var h_pool_flat = tf.reshape(h_pool, new TensorShape(-1, num_filters * filter_sizes.Rank));
            Tensor h_drop = null;
            tf_with(tf.name_scope("dropout"), delegate
            {
                h_drop = tf.nn.dropout(h_pool_flat, keep_prob);
            });

            Tensor logits = null;
            Tensor predictions = null;
            tf_with(tf.name_scope("output"), delegate
            {
                logits = keras.layers.dense(h_drop, num_class);
                predictions = tf.argmax(logits, -1, output_type: tf.int32);
            });

            tf_with(tf.name_scope("loss"), delegate
            {
                var sscel = tf.nn.sparse_softmax_cross_entropy_with_logits(logits: logits, labels: y);
                var loss = tf.reduce_mean(sscel);
                var adam = tf.train.AdamOptimizer(learning_rate);
                var optimizer = adam.minimize(loss, global_step: global_step);
            });

            tf_with(tf.name_scope("accuracy"), delegate
            {
                var correct_predictions = tf.equal(predictions, y);
                var accuracy = tf.reduce_mean(tf.cast(correct_predictions, TF_DataType.TF_FLOAT), name: "accuracy");
            });
        }
        private IEnumerable<(NDArray, NDArray, int)> batch_iter(NDArray inputs, NDArray outputs, int batch_size, int num_epochs)
        {
            var num_batches_per_epoch = (len(inputs) - 1) / batch_size + 1;
            var total_batches = num_batches_per_epoch * num_epochs;
            foreach (var epoch in range(num_epochs))
            {
                foreach (var batch_num in range(num_batches_per_epoch))
                {
                    var start_index = batch_num * batch_size;
                    var end_index = Math.Min((batch_num + 1) * batch_size, len(inputs));
                    if (end_index <= start_index)
                        break;
                    yield return (inputs[new Slice(start_index, end_index)], outputs[new Slice(start_index, end_index)], total_batches);
                }
            }
        }
        private void Test()
        {
            var checkpoint = Path.Combine(dataDir, "word_cnn.ckpt-800");
            if (!File.Exists($"{checkpoint}.meta")) return;

            var graph = tf.Graph();
            using (var sess = tf.Session(graph))
            {
                var saver = tf.train.import_meta_graph($"{checkpoint}.meta");
                saver.restore(sess, checkpoint);

                Tensor x = graph.get_operation_by_name("x");
                Tensor y = graph.get_operation_by_name("y");
                Tensor is_training = graph.get_operation_by_name("is_training");
                Tensor accuracy = graph.get_operation_by_name("accuracy/accuracy");

                var batches = batch_iter(test_x, test_y, BATCH_SIZE, 1);
                float sum_accuracy = 0;
                int cnt = 0;
                foreach (var (batch_x, batch_y, total) in batches)
                {
                    float accuracy_out = sess.run(accuracy, (x, batch_x), (y, batch_y), (is_training, false));
                    sum_accuracy += accuracy_out;
                    cnt += 1;
                }
                print($"Test Accuracy : {sum_accuracy / cnt}");
            }
        }
        private void Predict()
        {
            var checkpoint = Path.Combine(dataDir, "word_cnn.ckpt-800");
            if (!File.Exists($"{checkpoint}.meta")) return;

            var graph = tf.Graph();
            using (var sess = tf.Session(graph))
            {
                var saver = tf.train.import_meta_graph($"{checkpoint}.meta");
                saver.restore(sess, checkpoint);

                Tensor x = graph.get_operation_by_name("x");
                Tensor is_training = graph.get_operation_by_name("is_training");
                Tensor prediction = graph.get_operation_by_name("output/ArgMax");
                int test_num = 0;
                var test_contents = File.ReadAllLines(TEST_PATH);
                var (test_x, test_y) = build_word_dataset(TEST_PATH, word_dict, WORD_MAX_LEN);
                var input = ((NDArray)test_x[test_num]).reshape(1, 100);
                var result = sess.run(prediction, (x, input), (is_training, false));
                print($"Sentence: {test_contents[test_num]}");
                print($"Real: {test_y[test_num] + 1}");
                print($"Prediction Result: {result + 1}");
            }
        }
    }
}
