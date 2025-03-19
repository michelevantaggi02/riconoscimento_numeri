using OpenCvSharp;
using SkiaSharp;
using YoloDotNet.Models;

namespace riconoscimento_numeri.classes
{
    public class YoloDetection
    {

        public List<ObjectDetection> Detections { get; init; }
        public String ImagePath { get; init; }

        public Mat Image { get; init; }

        public YoloDetection()
        {
            // Costruttore
        }

        public Mat[] GetCuts()
        {
            Mat[] mats = new Mat[Detections.Count];
            for (int i = 0; i < Detections.Count; i++)
            {
                mats[i] = GetCut(i);
            }

            return mats;
        }

        public Mat GetCut(int index)
        {
            var detection = Detections[index];
            var cut = Image.SubMat(GetBounds(detection));
            return cut;
        }

        public static Rect GetBounds(ObjectDetection detection)
        {
            SKRectI bounds = detection.BoundingBox;

            int left = Math.Max(bounds.Left, 0);
            int top = Math.Max(bounds.Top, 0);

            return new Rect(left, top, bounds.Width, bounds.Height);
        }

        public Mat Cut(Rect bounds)
        {

            var cut = Image.SubMat(bounds.Top, bounds.Bottom, bounds.Left, bounds.Right);
            return cut;
        }



    }
}
