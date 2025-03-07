using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace riconoscimento_numeri.classes
{
    public class ManipolazioneImmagini
    {
        public static Mat Threshold(Mat image)
        {

            Mat hsv = image.CvtColor(ColorConversionCodes.BGR2HSV);

            InputArray lower_bound = InputArray.Create([0, 0, 135]);
            InputArray upper_bound = InputArray.Create([179, 62, 255]);



            return hsv.InRange(lower_bound, upper_bound);
        }

        public static Mat[] FindSquares(Mat image)

            
        {
            Mat[] squares = CalcEpsilon(image);
            List<Mat> cuts = [];
            foreach (Mat square in squares)
            {

                Mat checkMat = square;
                if (((double)square.Height) / ((double)square.Width) > 1.1)
                {
                    int s = square.Height / 3;
                    Rect changedRect = new Rect(0, s, square.Width, square.Height - (s));

                    checkMat = new Mat(square, changedRect);
                }



                Mat result = ApplyFilters(checkMat);

                cuts.Add(result);
            }
            return [..cuts];
        }

        public static Mat RemoveImperfections(Mat image)
        {
            Cv2.BitwiseNot(image, image);

            Cv2.FindContours(image, out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxNone);

            ConcurrentBag<Point[]> cleared = new();
            Parallel.ForEach(contours, (cont) =>
            {
                Point[]? clearedCont = RemoveSmall(cont);
                if (clearedCont != null)
                {
                    cleared.Add(clearedCont);
                }
            });

            image.DrawContours(cleared, -1, Scalar.Black, -1);

            Cv2.BitwiseNot(image, image);

            return image;
        }
        private static Mat[] CalcEpsilon(Mat thres, double e = 0.07110000000000001)
        {

            thres.FindContours(out Point[][] contours, out _, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            List<Mat> approxed = [];

            foreach (Point[] cont in contours)
            {
                double epsilon = e * Cv2.ArcLength(cont, false);

                Point[] approx = Cv2.ApproxPolyDP(cont, epsilon, true);


                

                RotatedRect areaRect = Cv2.MinAreaRect(approx);
                Rect bounds = Cv2.BoundingRect(approx);
                double area = Cv2.ContourArea(approx);

                if (bounds.Width > 30 && bounds.Width < 100 && bounds.Height > 20 && bounds.Height < 100 && area > 600)
                {
                    Mat cut = new Mat(thres, bounds);

                    Mat rotationMat = Cv2.GetRotationMatrix2D(new Point2f(bounds.Width / 2, bounds.Height / 2), - (areaRect.Angle % 10) / 2, 1);

                    Mat result = cut.WarpAffine(rotationMat, new Size(bounds.Width, bounds.Height), InterpolationFlags.Nearest, borderValue: Scalar.White);

                    approxed.Add(result);
                }
            }
            return [.. approxed];

        }

        private static Mat ApplyFilters(Mat image)
        {
            Mat blurred = new();
            Cv2.GaussianBlur(image, blurred, new Size(5, 5), 0);

            CLAHE cLAHE = Cv2.CreateCLAHE(2.0, new Size(5, 5));
            Mat enhanced = new();

            cLAHE.Apply(blurred, enhanced);

            Mat enhThres = new();
            Cv2.Threshold(enhanced, enhThres, 0, 255, ThresholdTypes.Otsu);

            return ClearImage(enhThres);
        }

        private static Mat ClearImage(Mat image)
        {
            Cv2.BitwiseNot(image, image);
            image = ParallelClearContours(image);
            Cv2.BitwiseNot(image, image);
            return image;
        }

        private static Mat ParallelClearContours(Mat image)
        {
            image.FindContours(out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxNone);

            
            ConcurrentBag<Point[]> cleared = new();

            Parallel.ForEach(contours, (cont) =>
            {
                Point[]? clearedCont = CheckPoints(cont, image.Width, image.Height);
                if (clearedCont != null)
                {
                    cleared.Add(clearedCont);
                }
            });

            image.DrawContours(cleared, -1, Scalar.Black, -1);

            return image;
        }

        private static Point[]? RemoveSmall(Point[] c)
        {
            double area = Cv2.ContourArea(c);
            if (area < 85 || area > 550)
            {
                return c;
            }
            else
            {
                return null;
            }
        }

        private static Point[]? CheckPoints(Point[] c, int W, int H)
        {
            Point[]? small = RemoveSmall(c);

            if (small != null)
            {
                return small;
            }
            else
            {
                bool Eliminable = false;
                double DistanzaMinima = Double.MaxValue;

                foreach (Point p in c)
                {

                    //Console.WriteLine($"{p.X}, {p.Y}");
                    if (p.X < 2 || p.Y < 2 || p.X >= W - 1)
                    {
                        Eliminable = true;
                    }
                    else
                    {
                        double dist = Math.Sqrt(Math.Pow(p.X - (W / 2), 2) + Math.Pow(p.Y - (H / 2), 2));
                        if (dist < DistanzaMinima)
                        {
                            DistanzaMinima = dist;
                        }
                    }
                }

                if(Eliminable && DistanzaMinima > 17)
                {
                    return c;
                }
            }



            return null;
        }
    }
}
