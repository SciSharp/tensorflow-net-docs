using System;

namespace YoloV3_train
{
    class Program
    {
        static void Main(string[] args)
        {
            var yolo3 = new SampleYOLOv3();
            yolo3.Run();

            Console.WriteLine("YOLOv3 is completed.");
            Console.ReadLine();
        }
    }
}
