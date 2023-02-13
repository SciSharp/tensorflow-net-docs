﻿using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace YoloV3_train.Core
{
    class Common
    {
        public static Tensor convolutional(Tensor input_layer, TensorShape filters_shape,
            bool downsample = false, bool activate = true,
            bool bn = true)
        {
            int strides;
            string padding;

            if (downsample)
            {
                var zero_padding_2d = keras.layers.ZeroPadding2D(new[,] { { 1, 0 }, { 1, 0 } });
                input_layer = zero_padding_2d.Apply(input_layer);
                strides = 2;
                padding = "valid";
            }
            else
            {
                strides = 1;
                padding = "same";
            }

            var conv2d_layer = keras.layers.Conv2D(filters_shape[-1],
                kernel_size: filters_shape[0],
                strides: strides,
                padding: padding,
                use_bias: !bn,
                kernel_regularizer: keras.regularizers.l2(0.0005f),
                kernel_initializer: tf.random_normal_initializer(stddev: 0.01f),
                bias_initializer: tf.constant_initializer(0f));
            var conv = conv2d_layer.Apply(input_layer);
            if (bn)
            {
                var batch_layer = keras.layers.BatchNormalization();
                conv = batch_layer.Apply(conv);
            }

            if (activate)
                conv = keras.layers.LeakyReLU(alpha: 0.1f).Apply(conv);

            return conv;
        }

        public static Tensor upsample(Tensor input_layer)
        {
            return keras.layers.UpSampling2D(size: (2, 2), interpolation: "nearest")
                .Apply(input_layer);
        }

        public static Tensor residual_block(Tensor input_layer,
            int input_channel, int filter_num1, int filter_num2)
        {
            var short_cut = input_layer;

            var conv = convolutional(input_layer, (1, 1, input_channel, filter_num1));
            conv = convolutional(conv, (3, 3, filter_num1, filter_num2));

            var residual_output = keras.layers.Add().Apply(new[] { short_cut, conv });

            return residual_output;
        }

    }
}
