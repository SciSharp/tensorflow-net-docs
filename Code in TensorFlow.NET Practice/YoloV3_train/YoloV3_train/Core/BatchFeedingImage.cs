using NumSharp;
using System.Collections.Generic;

namespace YoloV3_train.Core
{
    class BatchFeedingImage
    {
        public NDArray Image { get; set; }
        public List<LabelBorderBox> Targets { get; set; }
    }
}
