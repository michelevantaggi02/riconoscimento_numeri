
using SkiaSharp;
using Tesseract;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using static System.Net.Mime.MediaTypeNames;
using OpenCvSharp;
using System.ComponentModel;
using System.IO;
using System.Numerics;

namespace riconoscimento_numeri.classes
{
    internal class Riconoscimento
    {

        Yolo model = new Yolo(new YoloOptions
        {
            OnnxModel = @"models\yolov8m.onnx",
            ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
            Cuda = false,
        });



        public Riconoscimento()
        {

        }

        public void recognize(string path)
        {
            FileAttributes attributes = File.GetAttributes(path);

            if (attributes.HasFlag(FileAttributes.Directory))
            {
                recognize_directory(path);
            }
            else
            {
                recognize_image(path);
            }
        }

        public void recognize_directory(string dir_path)
        {
            FileAttributes attributes = File.GetAttributes(dir_path);

            if (!attributes.HasFlag(FileAttributes.Directory))
            {
                throw new Exception("Path is file, use recognize_image()");
            }

            var images = Directory.GetFiles(dir_path);

            Parallel.ForEach(images, recognize_image);
        }

        public void recognize_image(string image_path)
        {
            FileAttributes attributes = File.GetAttributes(image_path);

            if (attributes.HasFlag(FileAttributes.Directory))
            {
                throw new Exception("Path is directory, use recognize_directory()");
            }

            using var image = SKImage.FromEncodedData(image_path);
            var result = model.RunObjectDetection(image);

            save_result(image, image_path, result);

        }

        private void save_result(SKImage image, string path, List<ObjectDetection> result)
        {

            string name = Path.GetFileName(path);
            string dir = Path.GetDirectoryName(path) ?? "";

            List<string> labels = [];

            foreach (var item in result)
            {
                if (!labels.Contains(item.Label.Name))
                {
                    labels.Add(item.Label.Name);
                }

            }

            labels.ForEach(Console.WriteLine);

            //result = result.Where(x => x.Confidence > 0.7 && x.Label.Name.Equals("car")).ToList();

            using var resultImage = image.Draw(result);

            string full_img_path = Path.Combine(dir, @"full\");

            Directory.CreateDirectory(full_img_path);

            resultImage.Save(Path.Combine(full_img_path, name), SKEncodedImageFormat.Jpeg, 80);




            string cuts_img_path = Path.Combine(dir, @"cuts\", name);

            Directory.CreateDirectory(cuts_img_path);


            using var engine = new TesseractEngine(@"models", "ita", EngineMode.Default);

            for (int i = 0; i < result.Count; i++)
            {
                var item = result[i];
                using var temp = image.Subset(item.BoundingBox);


                //using var filtered = filter_image(temp);

                string temp_path = Path.Combine(cuts_img_path, i + ".jpg");
                temp.Save(temp_path, SKEncodedImageFormat.Jpeg, 80);

                recognize_number(temp_path);


            }

        }

        private Mat threshold( Mat image)
        {
            
            Mat hsv = image.CvtColor(ColorConversionCodes.BGR2HSV);

            InputArray lower_bound = InputArray.Create([98, 0, 127]);
            InputArray upper_bound = InputArray.Create([155, 90, 255]);



            return hsv.InRange(lower_bound, upper_bound);
        }

        private Mat[] calc_epsilon(Mat thres, Mat image, double e)
        { 

            thres.FindContours(out Point[][] contours, out _, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            List<Mat> approxed = [];

            foreach (Point[] p in contours)
            {
                double epsilon = e * Cv2.ArcLength(p, false);

                Point[] approx = Cv2.ApproxPolyDP(p, epsilon, true);
                OpenCvSharp.Rect bounds = Cv2.BoundingRect(approx);

                if (bounds.Width > 20 && bounds.Height > 20 && approx.Length == 3)
                {

                    Mat cut = new Mat(image, bounds);

                    
                    approxed.Add(cut);
                }
            }

            return [.. approxed];


        }

        private Mat[] test_filter(string path)
        {
            Mat image = Cv2.ImRead(path, ImreadModes.Color);
            Cv2.NamedWindow("Cut");
            Cv2.ImShow("Cut", image);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();


            /* Mat threshes = new Mat();
             List<Mat> thress = [];

             for(int i = 100; i<= 180; i += 10)
             {
                 Console.WriteLine(i);
                 Mat thres = inverted.Threshold(i, 255, ThresholdTypes.Binary);

                   thress.Add(thres);
             }
             if (thress.Count > 0)
             {
                 Cv2.HConcat(thress, threshes);
                 Cv2.NamedWindow("threshold " + path);

                 Cv2.ImShow("threshold " + path, threshes);

                 Cv2.WaitKey();
                 Cv2.DestroyWindow("threshold " + path);
             }*/

            Mat thres = threshold(image); //image.AdaptiveThreshold(255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 15, 3);
            return calc_epsilon(thres, image,  0.07110000000000001);

            //thres = thres.CvtColor(ColorConversionCodes.GRAY2RGB);
            //thres.DrawContours(contours, -1, Scalar.Red);
            //thres.DrawContours(approx, -1, Scalar.Green);
            Cv2.NamedWindow("threshold " + path);

            Cv2.ImShow("threshold " + path, thres);

            Cv2.WaitKey();
            Cv2.DestroyWindow("threshold " + path);




            Cv2.ImWrite(path, thres);
        }

        private SKImage filter_image(SKImage image)
        {

            var bitmap = SKBitmap.FromImage(image);            

            var img_info = new SKImageInfo(image.Width, image.Height); 
            using var surface = SKSurface.Create(img_info);

            var canvas = surface.Canvas;

            canvas.Clear(SKColors.Transparent);

            var grayColorFilter = SKColorFilter.CreateHighContrast(new SKHighContrastConfig
            {
                Grayscale = true,
                Contrast = .4f,
                InvertStyle = SKHighContrastConfigInvertStyle.InvertLightness,

            });

            var paint = new SKPaint
            {
                ColorFilter = grayColorFilter,
            };

            canvas.DrawBitmap(bitmap, 0, 0, paint);


            var snap = surface.Snapshot();

            return snap;

        }

        
        private void recognize_number(string cut_path)
        {


            //can't use a single engine with multithreading
            using TesseractEngine engine = new(@"models", "ita", EngineMode.Default);

            engine.SetVariable("tessedit_char_whitelist", "0123456789");


            Mat[] images = test_filter(cut_path);

            Console.WriteLine("Recognizing text");

            foreach (Mat image in images)
            {

                using Pix pixImage = Pix.LoadFromMemory(image.ToBytes());

                using var page = engine.Process(pixImage);

                string text = page.GetText();

                Console.WriteLine($"Item: {text}");
                Cv2.NamedWindow("Cut");
                Cv2.ImShow("Cut", image);
                Cv2.WaitKey();
                Cv2.DestroyAllWindows();
            }


            
        }
    }
}
