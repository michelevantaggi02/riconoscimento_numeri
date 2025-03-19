
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Models;
using OpenCvSharp;

namespace riconoscimento_numeri.classes
{
    internal class RiconoscimentoYolo
    {

        public Yolo model;



        public RiconoscimentoYolo(string yoloPath = @"models\yolov8m.onnx")
        {
            model = new Yolo(new YoloOptions
            {
                OnnxModel = yoloPath,
                ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
                Cuda = true,
            });
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



            }



            return [.. detections];
        }
    }
}
