using System;

namespace YOLOv3_Test
{
    class Program
    {
        static void Main(string[] args)
        {
            //var test = new TestImage();
            var test = new TestVideo();
            test.Run();

            Console.WriteLine("YOLOv3 Test is completed.");
            Console.ReadLine();
        }
    }
}
