using System.Collections.Concurrent;
using OpenCvSharp;

namespace riconoscimento_numeri.classes
{
    public class ImageManipulation
    {
        /// <summary>
        /// Turns the image into a mask keeping only white values
        /// </summary>
        /// <param name="image">original image in BGR format</param>
        /// <returns>Converted image</returns>
        public static Mat Threshold(Mat image)
        {

            Mat hsv = image.CvtColor(ColorConversionCodes.BGR2HSV);

            InputArray lower_bound = InputArray.Create([0, 0, 135]);
            InputArray upper_bound = InputArray.Create([179, 62, 255]);



            return hsv.InRange(lower_bound, upper_bound);
        }


        /// <summary>
        /// Checks for white squares in the image
        /// </summary>
        /// <param name="image">Thresholded image</param>
        /// <returns>List of squares resembling the sticker</returns>
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
                /*int borderSize = 15;
                result = result.CopyMakeBorder(borderSize, borderSize, borderSize, borderSize, BorderTypes.Constant, Scalar.White);*/

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

        /// <summary>
        /// Checks for white squares having dimensions corresponding to those of the sticker
        /// </summary>
        /// <param name="thres">Thresholded image</param>
        /// <param name="e">epsilon value for approximation</param>
        /// <returns>List of squares found</returns>
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

                //admitting only squares with a certain area and dimensions
                if (bounds.Width > 30 && bounds.Width < 100 && bounds.Height > 20 && bounds.Height < 100 && area > 600)
                {
                    Mat cut = new Mat(thres, bounds);

                    //rotating even a little helps Tesseract with recognition
                    Mat rotationMat = Cv2.GetRotationMatrix2D(new Point2f(bounds.Width / 2, bounds.Height / 2), - (areaRect.Angle % 10) / 2, 1);

                    Mat result = cut.WarpAffine(rotationMat, new Size(bounds.Width, bounds.Height), InterpolationFlags.Nearest, borderValue: Scalar.White);

                    approxed.Add(result);
                }
            }
            return [.. approxed];

        }


        /// <summary>
        /// Applies filters to the image to remove noise and clear from small imperfections
        /// </summary>
        /// <param name="image">Image to filter</param>
        /// <returns>Filtered Image</returns>
        private static Mat ApplyFilters(Mat image)
        {
            //Blur to obfuscate small imperfections
            Mat blurred = new();
            Cv2.GaussianBlur(image, blurred, new Size(5, 5), 0);


            //Local Histogram Equalization to remove noise
            CLAHE cLAHE = Cv2.CreateCLAHE(2.0, new Size(5, 5));
            Mat enhanced = new();

            cLAHE.Apply(blurred, enhanced);


            //Threshold to convert from grayscale to black and white only
            Mat enhThres = new();
            Cv2.Threshold(enhanced, enhThres, 0, 255, ThresholdTypes.Otsu);

            return ClearImage(enhThres);
        }

        /// <summary>
        /// Clears the image from black contours that doesn't resemble the numbers
        /// </summary>
        /// <param name="image">Image to clear</param>
        /// <returns>Image with unwanted borders removed</returns>
        public static Mat ClearImage(Mat image)
        {
            Cv2.BitwiseNot(image, image);
            image = ParallelClearContours(image);
            Cv2.BitwiseNot(image, image);
            return image;
        }

        /// <summary>
        /// Removes contours that doesn't resemble the numbers
        /// </summary>
        /// <param name="image"></param>
        /// <returns>Cleared Image</returns>
        private static Mat ParallelClearContours(Mat image)
        {
            image.FindContours(out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxNone);

            
            ConcurrentBag<Point[]> cleared = new();


            //Parallelizing the process
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

        /// <summary>
        /// Removes small and big contours
        /// </summary>
        /// <param name="c"></param>
        /// <returns>null if contour is ok, @c if contour area is not accepted</returns>
        private static Point[]? RemoveSmall(Point[] c)
        {
            double area = Cv2.ContourArea(c);
            if (area < 85 || area > 570)
            {
                return c;
            }
            else
            {
                return null;
            }
        }
        /// <summary>
        /// Checks if contours should be eliminated from image.
        /// Removes all contours that are too big or too small, and all contours that are too close to the image borders
        /// </summary>
        /// <param name="c">contour to check</param>
        /// <param name="W">image width</param>
        /// <param name="H">image heigth</param>
        /// <returns>null if contour is ok, @c if contour is to be removed</returns>
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
                double MinDistance = Double.MaxValue;

                foreach (Point p in c)
                {

                    if (p.X < 2 || p.Y < 2 || p.X >= W - 1)
                    {
                        Eliminable = true;
                    }
                    else
                    {
                        double dist = Math.Sqrt(Math.Pow(p.X - (W / 2), 2) + Math.Pow(p.Y - (H / 2), 2));
                        if (dist < MinDistance)
                        {
                            MinDistance = dist;
                        }
                    }
                }

                if(Eliminable && MinDistance > 17)
                {
                    return c;
                }
            }



            return null;
        }
    }
}
