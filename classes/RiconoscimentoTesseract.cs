using OpenCvSharp;
using Tesseract;
using YoloDotNet.Models;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Rect = OpenCvSharp.Rect;

namespace riconoscimento_numeri.classes
{
    public class RiconoscimentoTesseract
    {

        private class TesseractPrediction
        {
            public int Number { get; set; }
            public float Confidence { get; set; }
        }

        public static TesseractEngine engine = new TesseractEngine(@"C:\Users\michi\Desktop\riconoscimento_numeri\models\", "eng", EngineMode.Default);

    public static int[] Recognize(YoloDetection detection)
        {
            Mat thres = ManipolazioneImmagini.Threshold(detection.Image);

            /*Cv2.ImShow("thres", thres);
            int key = Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();*/
            int key = 0;

            //Console.WriteLine($"Detection count: {detection.Detections.Count}");

            List<int> numbers = [];


            foreach (ObjectDetection obj in detection.Detections)
            {
                Rect bounds = detection.GetBounds(obj);

                //Console.WriteLine($"Bounds converted: Left {bounds.Left}, Top {bounds.Top}, Width {bounds.Width}, Height {bounds.Height}");

                Rect cutRect = new(bounds.Left, bounds.Top, bounds.Width, bounds.Height);

                if (((double) bounds.Width) / ((double) bounds.Height) >= 2 && bounds.Left != 0)
                {
                    int remove_horiz = (bounds.Width / 3);
                    int remove_vert = (bounds.Height / 5);
                    
                    cutRect = new Rect(bounds.Left + remove_horiz, bounds.Top + remove_vert, bounds.Width - remove_horiz, bounds.Height - remove_vert);

                }

                //Console.WriteLine($"Rows: {cutRect.Top}-{cutRect.Bottom}, Cols: {cutRect.Left} - {cutRect.Right}");

                Mat cut = new(thres, cutRect);

                Mat[] squares = ManipolazioneImmagini.FindSquares(cut);
                int number = -1;
                float conf = 0.0f;

                //Console.WriteLine($"Squares: {squares.Length}");

                foreach (Mat square in squares)
                {

                    TesseractPrediction prediction = RunPrediction(square);
                    if (key == 13)
                    {
                        Console.WriteLine(prediction.Number);
                        Cv2.ImShow("square", square);
                        Cv2.WaitKey(0);
                        Cv2.DestroyAllWindows();
                    }

                    if (prediction.Confidence > conf)
                    {
                        number = prediction.Number;
                        conf = prediction.Confidence;
                    }
                }

                if (number == -1)
                {
                    //Console.WriteLine("Checking with different method");
                    Mat kernel = Mat.Ones(MatType.CV_8U, [1, 3]);

                    cut = cut.MorphologyEx(MorphTypes.Close, kernel);

                    cut = ManipolazioneImmagini.RemoveImperfections(cut);

                    Cv2.FindContours(cut, out var contours, out var hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxNone);

                    Mat conv = new();
                    Cv2.CvtColor(cut, conv, ColorConversionCodes.GRAY2RGB);

                    Cv2.DrawContours(conv, contours, -1, Scalar.Red, 1);

                    List<Point[]> filtered = [];

                    foreach (Point[] c in contours)
                    {
                        Rect rect = Cv2.BoundingRect(c);

                        Cv2.Rectangle(conv, rect, Scalar.Red, 1);

                        if ( ((double) rect.Height) /  ((double)rect.Width) > 1.2)
                        {
                            Cv2.Rectangle(conv, rect, Scalar.Blue, 2);
                            filtered.Add(c);
                        }


                    }
                    if (key == 13)
                    {
                        Cv2.ImShow("premerge", conv);
                        Cv2.WaitKey(0);
                        Cv2.DestroyAllWindows();
                    }
                    List<Point[]> merged = MergeContours([.. filtered]);

                    foreach(var p in merged)
                    {
                        Rect b = Cv2.BoundingRect(p);
                        Cv2.Rectangle(conv, b, Scalar.Green, 3);
                    }

                    if(key == 13)
                    {
                        Cv2.ImShow("square", conv);
                        Cv2.WaitKey(0);
                        Cv2.DestroyAllWindows();
                    }

                    while (number == -1 && merged.Count != 0)
                    {
                        Point[] largest = GetLargestContour([..merged]);

                        Rect rect = Cv2.BoundingRect(largest);

                        if (rect.Width > 20 && rect.Height > 20)
                        {
                            Mat cut2 = cut.SubMat(rect);

                            int borderSize = 15;

                            cut2 = cut2.CopyMakeBorder(borderSize, borderSize, borderSize, borderSize, BorderTypes.Constant, Scalar.White);

                            Mat dilateKernel = Mat.Ones(MatType.CV_8U, [2, 2]);

                            cut2 = cut2.Dilate(dilateKernel);

                            TesseractPrediction prediction = RunPrediction(cut2);
                            if (key == 13)
                            {
                                Console.WriteLine(prediction.Number);
                                Cv2.ImShow("square", cut2);
                                Cv2.WaitKey(0);
                                Cv2.DestroyAllWindows();
                            }

                            //Console.WriteLine($"Confidence: {prediction.Confidence}");
                            number = prediction.Number;
                        }

                        merged.Remove(largest);

                    }
                }

                //Console.WriteLine($"Number: {number}");

                numbers.Add(number);

            }

                return [..numbers];
        }

        private static TesseractPrediction RunPrediction(Mat image)
        {
            TesseractPrediction prediction = new() { Confidence = 0, Number = -1};
            
            engine.SetVariable("tessedit_char_whitelist", "0123456789");
            engine.DefaultPageSegMode = PageSegMode.SingleLine;
            using Pix pix = Pix.LoadFromMemory(image.ToBytes());

            using Page page = engine.Process(pix);
            
            string text = page.GetText();
            if (int.TryParse(text, out int found)){
                prediction.Number = found;
                prediction.Confidence = page.GetMeanConfidence();
            }

            return prediction;
        }

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

        private static bool AreClose(Point[] c1, Point[] c2, double threshold = 40)
        {
            Rect Rect1 = Cv2.BoundingRect(c1);
            Rect Rect2 = Cv2.BoundingRect(c2);


            bool L2R1 = Math.Abs(Rect2.Left - Rect1.Right) < threshold;
            bool L1R2 = Math.Abs(Rect2.Right - Rect1.Left) < threshold;

            bool L1L2 = Math.Abs(Rect2.Left - Rect1.Left) < threshold;
            bool R1R2 = Math.Abs(Rect2.Right - Rect1.Right) < threshold;


            bool horizontal = (L2R1 || L1R2) || (L1L2 || R1R2);

            if (horizontal)
            {
                return horizontal;
            }

            bool T1B2 = Math.Abs(Rect1.Top - Rect2.Bottom) < threshold;
            bool T2B1 = Math.Abs(Rect1.Bottom - Rect2.Top) < threshold;
            
            bool T1T2 = Math.Abs(Rect1.Top - Rect2.Top) < threshold;
            bool B1B2 = Math.Abs(Rect1.Bottom - Rect2.Bottom) < threshold;

            bool vertical = (T1B2 || T2B1) || (T1T2 || B1B2);

            return vertical;
        }

        private static List<Point[]> MergeContours(List<Point[]> contours, double threshold = 40)
        {
            bool changes = true;
            bool[] taken = new bool[contours.Count];


            while (changes)
            {
                Console.WriteLine($"Contours length: {contours.Count}");
                changes = false;
                List<Point[]> mergedContours = [];
                taken = new bool[contours.Count];

                for (int i = 0; i < contours.Count; i++)
                {
                    if (taken[i])
                    {
                        Console.WriteLine(contours[i]);
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
                                Console.WriteLine($"Merging 2 contours..");

                                newContour = [.. newContour, .. next];
                                Console.WriteLine($"{current.Length}'+ {next.Length} = {newContour.Length}");
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
