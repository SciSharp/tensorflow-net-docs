using System;
using NumSharp;
using Tensorflow;
using static SharpCV.Binding;
using static YOLOv3_Test.YOLOv3;

namespace YOLOv3_Test
{
    class TestVideo
    {
        public bool Run()
        {
            PredictFromVideo();
            return true;
        }
        private void PredictFromVideo()
        {
            var graph = ImportGraph();
            using (var sess = new Session(graph))
            {
                // Opens MP4 file (ffmpeg is probably needed)
                var vid = cv2.VideoCapture(AppDomain.CurrentDomain.BaseDirectory + @"yolov3\road.mp4");
                int sleepTime = (int)Math.Round(1000 / 24.0);
                var (loaded, frame) = vid.read();
                while (loaded)
                {
                    var frame_size = (frame.shape[0], frame.shape[1]);
                    var image_data = image_preporcess(frame, (input_size, input_size));
                    image_data = image_data[np.newaxis, Slice.Ellipsis];

                    var (pred_sbbox, pred_mbbox, pred_lbbox) = sess.run((return_tensors[1], return_tensors[2], return_tensors[3]),
                        (return_tensors[0], image_data));

                    var pred_bbox = np.concatenate((np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))), axis: 0);

                    var bboxes = postprocess_boxes(pred_bbox, frame_size, input_size, 0.3f);
                    var bboxess = nms(bboxes, 0.45f, method: "nms");
                    var image = draw_bbox(frame, bboxess);

                    cv2.imshow("objects", image);
                    cv2.waitKey(sleepTime);

                    (loaded, frame) = vid.read();
                }
            }
        }
    }
}
