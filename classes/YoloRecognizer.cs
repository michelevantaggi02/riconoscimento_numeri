
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.Models;
using OpenCvSharp;

namespace riconoscimento_numeri.classes
{
    /// <summary>
    /// YoloDotNet implementation.
    /// </summary>
    internal class YoloRecognizer
    {
        
        public Yolo model;



        public YoloRecognizer(string yoloPath = @"models\yolov8m.onnx")
        {
            model = new Yolo(new YoloOptions
            {
                OnnxModel = yoloPath,
                ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
                Cuda = true,
            });
        }


        /// <summary>
        /// Automatically recognizes a single image or a directory of images
        /// </summary>
        /// <param name="path"></param>
        /// <returns>Array of detections</returns>
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

        /// <summary>
        /// Recognizes all images in a directory
        /// </summary>
        /// <param name="dir_path"></param>
        /// <returns>Array of detections</returns>
        /// <exception cref="Exception">Given path is a file not a directory</exception>
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

        /// <summary>
        /// Recognizes a single image file
        /// </summary>
        /// <param name="image_path"></param>
        /// <returns>Detection informations</returns>
        /// <exception cref="Exception">Given path is not a file</exception>
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

            //We don't need the alpha channel
            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);

            List<ObjectDetection> detections = result.Where(x => x.Label.Name.Equals("car") || x.Label.Name.Equals("truck")).ToList();

            detections = filterIntersecting(detections);

            return new YoloDetection
            {
                Detections = detections,
                Image = mat,
            };

        }

        /// <summary>
        /// Recognizes a video file, not recommended since it's slow
        /// </summary>
        /// <param name="video_path"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
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
    
        private List<ObjectDetection> filterIntersecting(List<ObjectDetection> detections)
        {
            List<ObjectDetection> filtered = new();
            bool[] ignored = new bool[detections.Count];
            for (int i = 0; i < detections.Count; i++)
            {

                for (int j = i + 1; j < detections.Count; j++)
                {
                    
                    Rect bound1 = YoloDetection.GetBounds(detections[i]);
                    Rect bound2 = YoloDetection.GetBounds(detections[j]);


                    Rect intersection = bound1.Intersect(bound2);

                    if((intersection.Width > bound1.Width * 0.5 && intersection.Height > bound1.Height * 0.5) || (intersection.Width > bound2.Width * 0.5 && intersection.Height > bound2.Height * 0.5))
                    {
                        int area1 = bound1.Width * bound1.Height;
                        int area2 = bound2.Width * bound2.Height;

                        if (area1 > area2)
                        {
                            ignored[j] = true;
                        }
                        else
                        {
                            ignored[i] = true;
                        }

                    }
                }
                if (!ignored[i])
                {
                    filtered.Add(detections[i]);
                }
            }
            return filtered;
        }
    }

}
