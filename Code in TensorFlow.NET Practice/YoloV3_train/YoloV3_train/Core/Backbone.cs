﻿using Tensorflow;
using static Tensorflow.Binding;


namespace YoloV3_train.Core
{
    class Backbone
    {

        public static (Tensor, Tensor, Tensor) darknet53(Tensor input_data)
        {
            input_data = Common.convolutional(input_data, (3, 3, 3, 32));
            input_data = Common.convolutional(input_data, (3, 3, 32, 64), downsample: true);
            foreach (var i in range(1))
                input_data = Common.residual_block(input_data, 64, 32, 64);

            input_data = Common.convolutional(input_data, filters_shape: (3, 3, 64, 128), downsample: true);
            foreach (var i in range(2))
                input_data = Common.residual_block(input_data, 128, 64, 128);

            input_data = Common.convolutional(input_data, filters_shape: (3, 3, 128, 256), downsample: true);

            foreach (var i in range(8))
                input_data = Common.residual_block(input_data, 256, 128, 256);

            var route_1 = input_data;
            input_data = Common.convolutional(input_data, filters_shape: (3, 3, 256, 512), downsample: true);

            foreach (var i in range(8))
                input_data = Common.residual_block(input_data, 512, 256, 512);

            var route_2 = input_data;
            input_data = Common.convolutional(input_data, filters_shape: (3, 3, 512, 1024), downsample: true);

            foreach (var i in range(4))
                input_data = Common.residual_block(input_data, 1024, 512, 1024);

            return (route_1, route_2, input_data);
        }

    }
}
