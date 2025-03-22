using OpenCvSharp;
using SkiaSharp;
using YoloDotNet.Models;

namespace riconoscimento_numeri.classes
{

    /// <summary>
    /// Result of running a single image with Yolo
    /// </summary>
    public class YoloDetection()
    {

        /// <summary>
        /// List of detections with their bounding boxes
        /// </summary>
        public List<ObjectDetection> Detections { get; init; }

        /// <summary>
        /// OpenCV image object
        /// </summary>
        public Mat Image { get; init; }


        /// <summary>
        /// Returns subset of image of every detections
        /// </summary>
        /// <returns></returns>
        public Mat[] GetCuts()
        {
            Mat[] mats = new Mat[Detections.Count];
            for (int i = 0; i < Detections.Count; i++)
            {
                mats[i] = GetCut(i);
            }

            return mats;
        }

        /// <summary>
        /// Returns a cut of the image based on the detection index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Mat GetCut(int index)
        {
            var detection = Detections[index];
            var cut = Image.SubMat(GetBounds(detection));
            return cut;
        }

        /// <summary>
        /// Converts detection bounding box to OpenCV Rect
        /// </summary>
        /// <param name="detection"></param>
        /// <returns></returns>
        public static Rect GetBounds(ObjectDetection detection)
        {
            SKRectI bounds = detection.BoundingBox;

            int left = Math.Max(bounds.Left, 0);
            int top = Math.Max(bounds.Top, 0);

            return new Rect(left, top, bounds.Width, bounds.Height);
        }




    }
}
