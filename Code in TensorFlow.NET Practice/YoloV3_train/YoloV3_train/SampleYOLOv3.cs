﻿using NumSharp;
using YoloV3_train.Core;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static SharpCV.Binding;
using SharpCV;
using Utils = YoloV3_train.Core.Utils;

namespace YoloV3_train
{
    class SampleYOLOv3
    {
        YOLOv3 yolo;
        YoloDataset trainset, testset;
        YoloConfig cfg;

        OptimizerV2 optimizer;
        IVariableV1 global_steps;
        int warmup_steps;
        int total_steps;
        Tensor lr_tensor;

        Model model;
        int INPUT_SIZE = 416;

        public bool Run()
        {
            cfg = new YoloConfig("YOLOv3");
            yolo = new YOLOv3(cfg);

            PrepareData();
            //Train();
            Test();

            return true;
        }

        /// <summary>
        /// Train model in batch image
        /// </summary>
        /// <param name="image_data"></param>
        /// <param name="targets"></param>
        Tensor TrainStep(NDArray image_data, List<LabelBorderBox> targets)
        {
            using var tape = tf.GradientTape();
            var pred_result = model.Apply(image_data, training: true);
            var giou_loss = tf.constant(0.0f);
            var conf_loss = tf.constant(0.0f);
            var prob_loss = tf.constant(0.0f);

            // optimizing process in different border boxes.
            foreach (var (i, target) in enumerate(targets))
            {
                var (conv, pred) = (pred_result[i * 2], pred_result[i * 2 + 1]);
                var loss_items = yolo.compute_loss(pred, conv, target.Label, target.BorderBox, i);
                giou_loss += loss_items[0];
                conf_loss += loss_items[1];
                prob_loss += loss_items[2];
            }

            var total_loss = giou_loss + conf_loss + prob_loss;

            var gradients = tape.gradient(total_loss, model.trainable_variables);
            optimizer.apply_gradients(zip(gradients, model.trainable_variables.Select(x => x as ResourceVariable)));
            float lr = optimizer.lr.numpy();
            print($"=> STEP {global_steps.numpy():D4} lr:{lr} giou_loss: {giou_loss.numpy()} conf_loss: {conf_loss.numpy()} prob_loss: {prob_loss.numpy()} total_loss: {total_loss.numpy()}");
            global_steps.assign_add(1);

            // update learning rate
            int global_steps_int = global_steps.numpy();
            if (global_steps_int < warmup_steps)
            {
                lr = global_steps_int / (warmup_steps + 0f) * cfg.TRAIN.LEARN_RATE_INIT;
            }
            else
            {
                lr = (cfg.TRAIN.LEARN_RATE_END + 0.5f * (cfg.TRAIN.LEARN_RATE_INIT - cfg.TRAIN.LEARN_RATE_END) *
                    (1 + tf.cos((global_steps_int - warmup_steps + 0f) / (total_steps - warmup_steps) * (float)np.pi))).numpy();
            }
            lr_tensor = tf.constant(lr);
            optimizer.lr.assign(lr_tensor);

            return total_loss;
        }

        public void Train()
        {
            var input_layer = keras.layers.Input((416, 416, 3));
            var conv_tensors = yolo.Apply(input_layer);

            var output_tensors = new Tensors();
            foreach (var (i, conv_tensor) in enumerate(conv_tensors))
            {
                var pred_tensor = yolo.Decode(conv_tensor, i);
                output_tensors.Add(conv_tensor);
                output_tensors.Add(pred_tensor);
            }

            model = keras.Model(input_layer, output_tensors);
            model.summary();

            model.load_weights("./YOLOv3/yolov3.mnist.pretrain.h5");

            optimizer = keras.optimizers.Adam();
            global_steps = tf.Variable(1, trainable: false);
            int steps_per_epoch = trainset.Length;
            total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch;
            warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch;

            float loss = -1;
            foreach (var epoch in range(cfg.TRAIN.EPOCHS))
            {
                print($"EPOCH {epoch + 1:D4}");
                foreach (var dataset in trainset)
                {
                    loss = TrainStep(dataset.Image, dataset.Targets).numpy();
                }
                model.save_weights($"./YOLOv3/yolov3.{loss:F2}.h5");
            }
        }

        public void Test()
        {
            var input_layer = keras.layers.Input((INPUT_SIZE, INPUT_SIZE, 3));
            var feature_maps = yolo.Apply(input_layer);

            var bbox_tensors = new Tensors();
            foreach (var (i, fm) in enumerate(feature_maps))
            {
                var bbox_tensor = yolo.Decode(fm, i);
                bbox_tensors.Add(bbox_tensor);
            }
            model = keras.Model(input_layer, bbox_tensors);

            model.load_weights("./YOLOv3/yolov3.mnist.trained.h5");

            var mAP_dir = Path.Combine("mAP", "ground-truth");
            Directory.CreateDirectory(mAP_dir);

            var annotation_files = File.ReadAllLines(cfg.TEST.ANNOT_PATH);
            foreach (var (num, line) in enumerate(annotation_files))
            {
                var annotation = line.Split(' ');
                var image_path = annotation[0];
                var image_name = image_path.Split(Path.DirectorySeparatorChar).Last();
                var original_image = cv2.imread(image_path);
                var image = cv2.cvtColor(original_image, ColorConversionCodes.COLOR_BGR2RGB);
                var count = annotation.Skip(1).Count();
                var bbox_data_gt = np.zeros((count, 5), np.int32);
                foreach (var (i, box) in enumerate(annotation.Skip(1)))
                {
                    bbox_data_gt[i] = np.array(box.Split(',').Select(x => int.Parse(x)));
                };
                var (bboxes_gt, classes_gt) = (bbox_data_gt[":", ":4"], bbox_data_gt[":", "4"]);

                print($"=> ground truth of {image_name}:");

                var bbox_mess_file = new List<string>();
                foreach (var i in range(bboxes_gt.shape[0]))
                {
                    var class_name = yolo.Classes[classes_gt[i]];
                    var bbox_mess = $"{class_name} {string.Join(" ", bboxes_gt[i].ToArray<int>())}";
                    bbox_mess_file.Add(bbox_mess);
                    print('\t' + bbox_mess);
                }

                var ground_truth_path = Path.Combine(mAP_dir, $"{num}.txt");
                File.WriteAllLines(ground_truth_path, bbox_mess_file);
                print($"=> predict result of {image_name}:");
                // Predict Process
                var image_size = image.shape.Dimensions.Take(2).ToArray();
                var image_data = Utils.image_preporcess(image, image_size).Item1;

                image_data = image_data[np.newaxis, Slice.Ellipsis];
                var pred_bbox = model.predict(image_data);
                pred_bbox = pred_bbox.Select(x => tf.reshape(x, new object[] { -1, tf.shape(x)[-1] })).ToList();
                var pred_bbox_concat = tf.concat(pred_bbox, axis: 0);
                var bboxes = Utils.postprocess_boxes(pred_bbox_concat.numpy(), image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD);
                if (bboxes.size > 0)
                {
                    var best_box_results = Utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method: "nms");
                    Utils.draw_bbox(image, best_box_results, yolo.Classes.Values.ToArray());
                    cv2.imwrite(Path.Combine(cfg.TEST.DECTECTED_IMAGE_PATH, Path.GetFileName(image_name)), image);
                }
            }
        }

        public void PrepareData()
        {
            string dataDir = Path.Combine("YOLOv3", "data");
            Directory.CreateDirectory(dataDir);

            trainset = new YoloDataset("train", cfg);
            testset = new YoloDataset("test", cfg);
        }
    }
}
