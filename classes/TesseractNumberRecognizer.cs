using OpenCvSharp;
using Tesseract;
using YoloDotNet.Models;
using Rect = OpenCvSharp.Rect;

namespace riconoscimento_numeri.classes
{
    public class TesseractNumberRecognizer
    {

        public static TesseractEngine engine = new(@"models\", "eng", EngineMode.Default);

        /// <summary>
        /// Recognizes car race number for every detection in the image
        /// </summary>
        /// <param name="detection">Yolo detection result</param>
        /// <returns>Every number recognized for every detection</returns>
        public static TesseractPrediction[][] Recognize(YoloDetection detection)
        {
            Mat thres = ImageManipulation.Threshold(detection.Image);

            List<TesseractPrediction[]> numbers = [];


            foreach (ObjectDetection obj in detection.Detections)
            {

                Rect bounds = YoloDetection.GetBounds(obj);


                Rect cutRect = new(bounds.Left, bounds.Top, bounds.Width, bounds.Height);

                if (((double)bounds.Width) / ((double)bounds.Height) >= 2 && bounds.Left != 0)
                {
                    int remove_horiz = (bounds.Width / 3);
                    int remove_vert = (bounds.Height / 5);

                    cutRect = new Rect(bounds.Left + remove_horiz, bounds.Top + remove_vert, bounds.Width - remove_horiz, bounds.Height - remove_vert);

                }

                Mat cut = new(thres, cutRect);

                Mat[] squares = ImageManipulation.FindSquares(cut);
                List<TesseractPrediction> predictions = [];


                // Checks if the square is a number
                foreach (Mat square in squares)
                {
                    TesseractPrediction prediction = RunPrediction(square);



                    if (prediction.Confidence > 0)
                    {
                        predictions.Add(prediction);
                    }
                }

                // Tries to find the number in a different way
                if (predictions.Count == 0)
                {
                    Mat kernel = Mat.Ones(MatType.CV_8U, [2, 3]);

                    cut = cut.MorphologyEx(MorphTypes.Close, kernel);

                    cut = ImageManipulation.RemoveImperfections(cut);

                    Cv2.FindContours(cut, out var contours, out var hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxNone);


                    List<Point[]> filtered = [];

                    foreach (Point[] c in contours)
                    {
                        Rect rect = Cv2.BoundingRect(c);

                        if (((double)rect.Height) / ((double)rect.Width) > 1.2)
                        {
                            filtered.Add(c);
                        }


                    }
                    List<Point[]> merged = MergeContours([.. filtered]);

                    foreach (var p in merged)
                    {
                        Rect b = Cv2.BoundingRect(p);
                    }

                    //Stops at the first number it finds
                    while (predictions.Count == 0 && merged.Count != 0)
                    {
                        Point[] largest = GetLargestContour([.. merged]);

                        Rect rect = Cv2.BoundingRect(largest);

                        if (rect.Width > 15 && rect.Height > 15)
                        {
                            Mat cut2 = cut.SubMat(rect);

                            int borderSize = 15;

                            cut2 = cut2.CopyMakeBorder(borderSize, borderSize, borderSize, borderSize, BorderTypes.Constant, Scalar.White);

                            //cut2 = ImageManipulation.ClearImage(cut2);

                            Mat dilateKernel = Mat.Ones(MatType.CV_8U, [2, 2]);

                            cut2 = cut2.Dilate(dilateKernel);

                            TesseractPrediction prediction = RunPrediction(cut2);

                            //Sometimes Tesseract returns a number with confidence 0
                            if (prediction.Confidence >= 0)
                            {
                                predictions.Add(prediction);
                            }
                        }

                        merged.Remove(largest);

                    }
                }


                numbers.Add([.. predictions]);

            }

            return [.. numbers];
        }

        /// <summary>
        /// Runs Tesseract prediction on prepared image
        /// </summary>
        /// <param name="image"></param>
        /// <returns>Prediction result</returns>
        private static TesseractPrediction RunPrediction(Mat image)
        {
            TesseractPrediction prediction = new() { Confidence = -1, Number = "NO" };

            engine.SetVariable("tessedit_char_whitelist", "0123456789");
            engine.DefaultPageSegMode = PageSegMode.SingleLine;
            using Pix pix = Pix.LoadFromMemory(image.ToBytes());

            using Page page = engine.Process(pix);

            string text = page.GetText();
            if (int.TryParse(text, out int found))
            {
                prediction.Number = found.ToString();
                prediction.Confidence = page.GetMeanConfidence();
            }

            return prediction;
        }

        /// <summary>
        /// Get contour with largest area
        /// </summary>
        /// <param name="contours"></param>
        /// <returns></returns>
        private static Point[] GetLargestContour(Point[][] contours)
        {
            Point[] largest = contours[0];
            double maxArea = Cv2.ContourArea(largest);
            foreach (Point[] cont in contours)
            {
                double area = Cv2.ContourArea(cont);
                if (area > maxArea)
                {
                    maxArea = area;
                    largest = cont;
                }
            }
            return largest;
        }

        /// <summary>
        /// Checks if 2 contours are close to each other
        /// </summary>
        /// <param name="c1"></param>
        /// <param name="c2"></param>
        /// <param name="threshold"> Maximum distance between points</param>
        /// <returns></returns>
        private static bool AreClose(Point[] c1, Point[] c2, double threshold = 10)
        {
            Rect Rect1 = Cv2.BoundingRect(c1);
            Rect Rect2 = Cv2.BoundingRect(c2);


            bool L2R1 = Math.Abs(Rect2.Left - Rect1.Right) < threshold;
            bool L1R2 = Math.Abs(Rect2.Right - Rect1.Left) < threshold;

            bool L1L2 = Math.Abs(Rect2.Left - Rect1.Left) < threshold;
            bool R1R2 = Math.Abs(Rect2.Right - Rect1.Right) < threshold;


            bool horizontal = (L2R1 || L1R2) || (L1L2 || R1R2);


            bool T1B2 = Math.Abs(Rect1.Top - Rect2.Bottom) < threshold;
            bool T2B1 = Math.Abs(Rect1.Bottom - Rect2.Top) < threshold;

            bool T1T2 = Math.Abs(Rect1.Top - Rect2.Top) < threshold;
            bool B1B2 = Math.Abs(Rect1.Bottom - Rect2.Bottom) < threshold;

            bool vertical = (T1B2 || T2B1) || (T1T2 || B1B2);

            return horizontal && vertical;
        }

        /// <summary>
        /// Merges contours that are close to each other
        /// </summary>
        /// <param name="contours"></param>
        /// <returns>New contour list</returns>
        private static List<Point[]> MergeContours(List<Point[]> contours)
        {
            bool changes = true;
            bool[] taken = new bool[contours.Count];


            while (changes)
            {
                changes = false;
                List<Point[]> mergedContours = [];
                taken = new bool[contours.Count];

                for (int i = 0; i < contours.Count; i++)
                {
                    if (taken[i])
                    {
                        continue;
                    }

                    taken[i] = true;

                    Point[] current = contours[i];
                    Point[] newContour = contours[i];

                    for (int j = 0; j < contours.Count; j++)
                    {
                        if (i != j && !taken[j])
                        {
                            Point[] next = contours[j];
                            if (AreClose(current, next))
                            {

                                newContour = [.. newContour, .. next];
                                taken[j] = true;
                                changes = true;
                            }
                        }

                    }
                    mergedContours.Add(newContour);

                }

                if (changes)
                {
                    contours = mergedContours;
                }
            }

            return contours;
        }


    }


}
