using static SharpCV.Binding;

namespace TensorDebuggerVisualizersTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var img = cv2.imread("img.bmp");
            var img_rotate = cv2.rotate(img, SharpCV.RotateFlags.ROTATE_90_CLOCKWISE);
        }
    }
}
