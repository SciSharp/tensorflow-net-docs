using System;
using NumSharp;
using SharpCV;
using Tensorflow;
using static SharpCV.Binding;
using static YOLOv3_Test.YOLOv3;

namespace YOLOv3_Test
{
    class TestImage
    {
        public bool Run()
        {
            PredictFromImage();
            return true;
        }
        private void PredictFromImage()
        {
            var graph = ImportGraph();
            using (var sess = new Session(graph))
            {
                var original_image_raw = cv2.imread(AppDomain.CurrentDomain.BaseDirectory + @"yolov3\cat_face.jpg");
                var original_image = cv2.cvtColor(original_image_raw, ColorConversionCodes.COLOR_BGR2RGB);
                var original_image_size = (original_image.shape[0], original_image.shape[1]);
                var image_data = image_preporcess(original_image, (input_size, input_size));
                image_data = image_data[np.newaxis, Slice.Ellipsis];

                var (pred_sbbox, pred_mbbox, pred_lbbox) = sess.run((return_tensors[1], return_tensors[2], return_tensors[3]),
                        (return_tensors[0], image_data));

                var pred_bbox = np.concatenate((np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))), axis: 0);

                var bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.03f);//as is: 0.3f
                var bboxess = nms(bboxes, 0.3f, method: "nms");//as is: 0.5f
                var image = draw_bbox(original_image_raw, bboxess);
                cv2.imshow("Detected Objects in TensorFlow.NET", image);
                cv2.waitKey();
            }
        }
    }
}