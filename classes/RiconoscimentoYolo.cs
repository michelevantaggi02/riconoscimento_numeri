
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
using System.Collections.Concurrent;

namespace riconoscimento_numeri.classes
{
    internal class RiconoscimentoYolo
    {

        public Yolo model = new Yolo(new YoloOptions
        {
            OnnxModel = @"models\yolov8m.onnx",
            ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
            Cuda = true,
        });



        public RiconoscimentoYolo()
        {

        }

        public YoloDetection[] recognize(string path)
        {
            FileAttributes attributes = File.GetAttributes(path);

            if (attributes.HasFlag(FileAttributes.Directory))
            {
                return recognize_directory(path);
            }
            else
            {
                return [recognize_image(path)];
            }
        }

        public YoloDetection[] recognize_directory(string dir_path)
        {
            FileAttributes attributes = File.GetAttributes(dir_path);

            if (!attributes.HasFlag(FileAttributes.Directory))
            {
                throw new Exception("Path is file, use recognize_image()");
            }

            var images = Directory.GetFiles(dir_path);

            List<YoloDetection> detections = new();

            foreach (var image in images) { 
                YoloDetection detection = recognize_image(image);
                detections.Add(detection);
            }

            return [.. detections];
        }

        public YoloDetection recognize_image(string image_path)
        {
            FileAttributes attributes = File.GetAttributes(image_path);

            if (attributes.HasFlag(FileAttributes.Directory))
            {
                throw new Exception("Path is directory, use recognize_directory()");
            }

            using var image = SKImage.FromEncodedData(image_path);
            var result = model.RunObjectDetection(image);

            using SKBitmap bitmap = SKBitmap.FromImage(image);

            SKPixmap pixmap = new();

            if (!bitmap.PeekPixels(pixmap))
            {
                throw new Exception("Impossibile ottenere i pixel");
            }

            IntPtr data = pixmap.GetPixels();

            Mat mat = Mat.FromPixelData(bitmap.Height, bitmap.Width, MatType.CV_8UC4, data);

            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);
            
            //save_result(image, image_path, result);

            return new YoloDetection
            {
                Detections = result.Where(x => x.Label.Name.Equals("car")).ToList(),
                Image = mat,
            };

        }

        public YoloDetection[] recognize_video(string video_path)
        {
            FileAttributes attributes = File.GetAttributes(video_path);
            if (attributes.HasFlag(FileAttributes.Directory))
            {
                throw new Exception("Path is directory, use recognize_directory()");
            }

            String output_dir = Path.Combine(Path.GetDirectoryName(video_path) ?? "", @"output\");

            Console.WriteLine(output_dir);

            Directory.CreateDirectory(output_dir);
            VideoCapture capture = new(video_path);

            VideoOptions videoOptions = new VideoOptions
            {
                VideoFile = video_path,
                OutputDir = output_dir,
                KeepAudio = false,
                KeepFrames = false,
                GenerateVideo = false,
                FPS = (float) capture.Fps,
            };



            var result = model.RunObjectDetection(videoOptions);
            List<YoloDetection> detections = new();



            foreach (var item in result)
            {

                capture.Set(VideoCaptureProperties.PosFrames, item.Key);
                
                Mat frame = new();
                capture.Read(frame);
                var recognition = item.Value.Where(x => x.Label.Name.Equals("car")).ToList(); 
                //string path = Path.Combine(output_dir, @"Temp\", $"{item.Key}.png");


                YoloDetection detection = new()
                {
                    Detections = recognition,
                    //ImagePath = path,
                    Image = frame,
                };

                detections.Add(detection);


                /*foreach (ObjectDetection detection in recognition)
                {

                    if(detection.Label.Name.Equals("car"))
                    {
                        Console.WriteLine("Car detected");


                        Console.WriteLine(path);
                        using var image = SKImage.FromEncodedData(path);
                        save_result(image, path, recognition);

                        break;
                    }
                }*/

            }



            return [.. detections];
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

            result = result.Where(x => x.Confidence > 0.7 && x.Label.Name.Equals("car")).ToList();

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

                recognize_number(@"C:\Users\michi\Desktop\riconoscimento_numeri\notebooks\test_audi.png");


            }

        }

        private Mat threshold( Mat image)
        {
            
            Mat hsv = image.CvtColor(ColorConversionCodes.BGR2HSV);

            InputArray lower_bound = InputArray.Create([98, 0, 127]);
            InputArray upper_bound = InputArray.Create([155, 90, 255]);



            return hsv.InRange(lower_bound, upper_bound);
        }

        private Mat[] calc_epsilon(Mat thres, double e = 0.07110000000000001)
        { 

            thres.FindContours(out Point[][] contours, out _, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

            List<Mat> approxed = [];

            foreach (Point[] cont in contours)
            {
                double epsilon = e * Cv2.ArcLength(cont, false);

                Point[] approx = Cv2.ApproxPolyDP(cont, epsilon, true);

                RotatedRect areaRect = Cv2.MinAreaRect(approx);
                OpenCvSharp.Rect bounds = Cv2.BoundingRect(approx);
                double area = Cv2.ContourArea(approx);

                if (bounds.Width > 30 && bounds.Width < 100 && bounds.Height > 20 && bounds.Height < 100 && area > 600)
                {

                    Mat cut = new Mat(thres, bounds);

                    Mat rotationMat = Cv2.GetRotationMatrix2D(new Point2f(bounds.Width / 2, bounds.Height / 2), areaRect.Angle, 1);

                    Mat result = cut.WarpAffine(rotationMat, new Size(bounds.Width, bounds.Height), InterpolationFlags.Nearest, borderValue: Scalar.White);

                    approxed.Add(result);
                }
            }

            return [.. approxed];


        }


        private Mat cleanImage(Mat image, float angle)
        {
            Mat equalized = equalizeImage(image);

            Mat eq_thres = equalized.Threshold(0, 255, ThresholdTypes.Otsu);

            Cv2.NamedWindow("Equalized Threshold", WindowFlags.KeepRatio);
            Cv2.ImShow("Equalized Threshold", eq_thres);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

            Mat morphed = morphImage(eq_thres);

            Mat cleaned = removeStains(morphed);

            Mat rotated = rotateImage(cleaned, angle);


            return rotated;
        }

        private Mat equalizeImage(Mat image)
        {
            Mat blurred = image.GaussianBlur(new Size(5, 5), 0);

            Mat equalized = blurred.EqualizeHist();

            Cv2.NamedWindow("Equalized", WindowFlags.KeepRatio);
            Cv2.ImShow("Equalized", equalized);

            CLAHE clahe = Cv2.CreateCLAHE(2.0, new Size(8, 8));
            Mat enhanced = new();
            clahe.Apply(equalized, enhanced);

            Cv2.NamedWindow("CLAHE", WindowFlags.KeepRatio);
            Cv2.ImShow("CLAHE", enhanced);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

            return enhanced;
        }

        private Mat morphImage(Mat image)
        {
            Mat kernel = Mat.Ones(MatType.CV_8U, [2, 2]);
            Mat dilated = image.MorphologyEx(MorphTypes.Close, kernel, iterations: 3);

            Cv2.NamedWindow("Morphology", WindowFlags.KeepRatio);
            Cv2.ImShow("Morphology", dilated);

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
            return dilated;
        }

        private Mat removeStains(Mat image)
        {
            Mat inv_copy = new();
            Cv2.BitwiseNot(image, inv_copy);

            Cv2.FindContours(inv_copy, out Point[][] macchie, out _, RetrievalModes.List, ContourApproximationModes.ApproxNone);

            Console.WriteLine(macchie.Length);

            foreach (Point[] macchia in macchie)
            {
                double area = Cv2.ContourArea(macchia);
                Console.WriteLine(area);

                if (area < 200)
                {
                    Cv2.DrawContours(inv_copy, [macchia], -1, Scalar.Black, -1);
                }
            }

            Cv2.NamedWindow("Senza macchie", WindowFlags.KeepRatio);
            Cv2.ImShow("Senza macchie", inv_copy);
            Cv2.BitwiseNot(inv_copy, inv_copy);

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

            return inv_copy;

        }

        private Mat rotateImage(Mat image, float angle)
        {
            Size bounds = image.Size();
            Point2f center = new Point2f(bounds.Width / 2, bounds.Height / 2);
            Mat rotation_mat = Cv2.GetRotationMatrix2D(center, angle, 1);

            Mat result = image.WarpAffine(rotation_mat, new Size(bounds.Width, bounds.Height), InterpolationFlags.Nearest, borderValue: Scalar.White);

            return result;
        }



        private Mat[] searchNumber(string path)
        {
            Mat image = Cv2.ImRead(path, ImreadModes.Color);

            Mat thres = threshold(image); //image.AdaptiveThreshold(255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 15, 3);

            return  calc_epsilon(thres,  0.07110000000000001);
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
            using TesseractEngine engine = new TesseractEngine(@"C:\Users\michi\Desktop\riconoscimento_numeri\models\", "eng", EngineMode.Default);

            Console.WriteLine(engine.Version);

            engine.SetVariable("tessedit_char_whitelist", "0123456789");
            engine.DefaultPageSegMode = PageSegMode.SingleLine;




            Mat[] images = searchNumber(cut_path);

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
