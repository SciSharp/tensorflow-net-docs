using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using SharpCV;
using System.IO;
using Tensorflow;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace YOLOv3_Test
{
    public static class YOLOv3
    {
        public static int input_size = 416;
        public static int num_classes = 80;
        public static Tensor[] return_tensors;
        static string[] return_elements = new[]{
            "input/input_data:0",
            "pred_sbbox/concat_2:0",
            "pred_mbbox/concat_2:0",
            "pred_lbbox/concat_2:0"
        };
        public static Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();

            var bytes = File.ReadAllBytes(AppDomain.CurrentDomain.BaseDirectory + @"yolov3\yolov3_coco.pb");
            var graphDef = GraphDef.Parser.ParseFrom(bytes);
            return_tensors = tf.import_graph_def(graphDef, return_elements: return_elements)
                .Select(x => x as Tensor)
                .ToArray();

            return graph;
        }
        public static NDArray image_preporcess(Mat image, (int, int) target_size)
        {
            image = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2RGB);
            var (ih, iw) = target_size;
            var (h, w) = (image.shape[0] + 0.0f, image.shape[1] + 0.0f);
            var scale = min(iw / w, ih / h);
            var (nw, nh) = ((int)Math.Round(scale * w), (int)Math.Round(scale * h));
            var image_resized = cv2.resize(image, (nw, nh));
            var image_padded = np.full((ih, iw, 3), fill_value: 128.0f);
            var (dw, dh) = ((iw - nw) / 2, (ih - nh) / 2);
            image_padded[new Slice(dh, nh + dh), new Slice(dw, nw + dw), Slice.All] = image_resized;
            image_padded = image_padded / 255;

            return image_padded;
        }
        public static NDArray postprocess_boxes(NDArray pred_bbox, (int, int) org_img_shape, float input_size, float score_threshold)
        {
            var valid_scale = new[] { 0, np.inf };

            var pred_xywh = pred_bbox[Slice.All, new Slice(0, 4)];
            var pred_conf = pred_bbox[Slice.All, 4];
            var pred_prob = pred_bbox[Slice.All, new Slice(5)];

            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var pred_coor = np.concatenate((pred_xywh[Slice.All, new Slice(stop: 2)] - pred_xywh[Slice.All, new Slice(2)] * 0.5f,
                                        pred_xywh[Slice.All, new Slice(stop: 2)] + pred_xywh[Slice.All, new Slice(2)] * 0.5f), axis: -1);

            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            var (org_h, org_w) = org_img_shape;
            var resize_ratio = min(input_size / org_w, input_size / org_h);
            var dw = (input_size - resize_ratio * org_w) / 2;
            var dh = (input_size - resize_ratio * org_h) / 2;

            pred_coor[Slice.All, new Slice(0, step: 2)] = 1.0 * (pred_coor[Slice.All, new Slice(0, step: 2)] - dw) / resize_ratio;
            pred_coor[Slice.All, new Slice(1, step: 2)] = 1.0 * (pred_coor[Slice.All, new Slice(1, step: 2)] - dh) / resize_ratio;

            // (3) clip some boxes those are out of range
            pred_coor = np.concatenate((np.maximum(pred_coor[Slice.All, new Slice(stop: 2)], np.array(new[] { 0, 0 })),
                np.minimum(pred_coor[Slice.All, new Slice(2)], np.array(new[] { org_w - 1, org_h - 1 }))), axis: -1);

            var invalid_mask = np.logical_or(pred_coor[Slice.All, 0] > pred_coor[Slice.All, 2], pred_coor[Slice.All, 1] > pred_coor[Slice.All, 3]);
            pred_coor[invalid_mask] = 0;

            // (4) discard some invalid boxes
            var coor_diff = pred_coor[Slice.All, new Slice(2, 4)] - pred_coor[Slice.All, new Slice(0, 2)];
            var bboxes_scale = np.sqrt(np.prod(coor_diff, axis: -1));
            var scale_mask = np.logical_and(bboxes_scale > 0d, bboxes_scale < double.MaxValue);

            // (5) discard some boxes with low scores
            NDArray coors;
            var classes = np.argmax(pred_prob, axis: -1);
            var scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes];
            var score_mask = scores > score_threshold;
            var mask = np.logical_and(scale_mask, score_mask);
            (coors, scores, classes) = (pred_coor[mask], scores[mask], classes[mask]);

            return np.concatenate(new[] { coors, scores[Slice.All, np.newaxis], classes[Slice.All, np.newaxis] }, axis: -1);
        }
        public static NDArray[] nms(NDArray bboxes, float iou_threshold, float sigma = 0.3f, string method = "nms")
        {
            var classes_in_img = bboxes[Slice.All, 5].Data<float>().Distinct().ToArray();
            var best_bboxes = new List<NDArray>();
            foreach (var cls in classes_in_img)
            {
                var cls_mask = bboxes[Slice.All, 5] == cls;
                var cls_bboxes = bboxes[cls_mask];
                while (len(cls_bboxes) > 0)
                {
                    var max_ind = np.argmax(cls_bboxes[Slice.All, 4]);
                    var best_bbox = cls_bboxes[max_ind];
                    best_bboxes.append(best_bbox);
                    cls_bboxes = np.concatenate(new[] { cls_bboxes[new Slice(stop: max_ind)], cls_bboxes[new Slice(max_ind + 1)] });
                    NDArray iou = bboxes_iou(best_bbox[np.newaxis, new Slice(stop: 4)], cls_bboxes[Slice.All, new Slice(stop: 4)]);

                    if (len(iou) == 0)
                        continue;

                    var weight = np.ones(new Shape(len(iou)), dtype: np.float32);

                    if (method == "nms")
                    {
                        var iou_mask = (iou > iou_threshold).MakeGeneric<bool>();
                        if (iou_mask.ndim == 0)
                            iou_mask = iou_mask.reshape(1);
                        if (iou_mask.size > 0)
                            weight[iou_mask] = 0.0f;
                    }
                    else if (method == "soft-nms")
                    {
                        weight = np.exp(-(1.0 * np.sqrt(iou) / sigma));
                    }

                    //if(len(cls_bboxes) > 0)
                    {
                        cls_bboxes[Slice.All, 4] = cls_bboxes[Slice.All, 4] * weight;
                        var score_mask = cls_bboxes[Slice.All, 4] > 0f;
                        cls_bboxes = cls_bboxes[score_mask];
                    }
                }
            }

            return best_bboxes.ToArray();
        }
        public static NDArray bboxes_iou(NDArray boxes1, NDArray boxes2)
        {
            if (boxes2.size == 0)
                return boxes2;

            var boxes1_area = (boxes1[Slice.Ellipsis, 2] - boxes1[Slice.Ellipsis, 0]) * (boxes1[Slice.Ellipsis, 3] - boxes1[Slice.Ellipsis, 1]);
            var boxes2_area = (boxes2[Slice.Ellipsis, 2] - boxes2[Slice.Ellipsis, 0]) * (boxes2[Slice.Ellipsis, 3] - boxes2[Slice.Ellipsis, 1]);

            var left_up = np.maximum(boxes1[Slice.Ellipsis, new Slice(stop: 2)], boxes2[Slice.Ellipsis, new Slice(stop: 2)]);
            var right_down = np.minimum(boxes1[Slice.Ellipsis, new Slice(2)], boxes2[Slice.Ellipsis, new Slice(2)]);

            var inter_section = np.maximum(right_down - left_up, 0.0);
            var inter_area = inter_section[Slice.Ellipsis, 0] * inter_section[Slice.Ellipsis, 1];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var ious = np.maximum(1.0 * inter_area / union_area, np.array(1.1920929e-7));

            return ious;
        }
        public static Mat draw_bbox(Mat image, NDArray[] bboxes)
        {
            // var rnd = new Random();
            var classes = File.ReadAllLines(AppDomain.CurrentDomain.BaseDirectory + @"yolov3\coco.names");
            var num_classes = len(classes);
            var (image_h, image_w) = (image.shape[0], image.shape[1]);
            // var hsv_tuples = range(num_classes).Select(x => (rnd.Next(255), rnd.Next(255), rnd.Next(255))).ToArray();

            foreach (var (i, bbox) in enumerate(bboxes))
            {
                var coor = bbox[new Slice(stop: 4)].astype(NPTypeCode.Int32);
                var fontScale = 0.5;
                float score = bbox[4];
                var class_ind = (float)bbox[5];
                var bbox_color = (0, 0, 250);// hsv_tuples[rnd.Next(num_classes)];
                var bbox_thick = (int)(0.6 * (image_h + image_w) / 600);
                cv2.rectangle(image, (coor[0], coor[1]), (coor[2], coor[3]), bbox_color, bbox_thick);

                // show label;
                var bbox_mess = $"{classes[(int)class_ind]}: {score.ToString("P")}";
                var t_size = cv2.getTextSize(bbox_mess, HersheyFonts.HERSHEY_SIMPLEX, fontScale, thickness: bbox_thick / 2);
                cv2.rectangle(image, (coor[0], coor[1]), (coor[0] + t_size.Width, coor[1] - t_size.Height - 3), bbox_color, -1);
                cv2.putText(image, bbox_mess, (coor[0], coor[1] - 2), HersheyFonts.HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick / 2, lineType: LineTypes.LINE_AA);
            }

            return image;
        }
    }
}
