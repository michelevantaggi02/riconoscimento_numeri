using OpenCvSharp;
using riconoscimento_numeri.classes;
using riconoscimento_numeri.classes.DeepSort;
using SkiaSharp;
using Tesseract;
using YoloDotNet.Models;


class Program {

    /// <summary>
    /// Reads every frame in the folder, applies DeepSort and Tesseract to recognize numbers
    /// </summary>
    /// <param name="args"></param>
    static void Main(string[] args) {


        Matcher DeepSortMatcher = new(@"models\yolov8m.onnx", @"models\FastReidVeRi.onnx");


        var watch = System.Diagnostics.Stopwatch.StartNew();

        List<YoloDetection> detections = new List<YoloDetection>();

        //For dimonstration purposes we will use a fixed number of frames
        int frames = 90;

        long fileReadTime = 0;
        long yoloTime = 0;
        long tesseractTime = 0;
        long drawTime = 0;

        List<Mat> images = [];

        HashSet<string> prevFrame = [];

        string framesPath = @"C:\Users\michi\Desktop\riconoscimento_numeri\vids\corti\frames\";
        string videoName = @"video1\";

        for (int i = 1; i <= frames; i++)
        {
            watch.Restart();

            //Recognize cars and apply DeepSort
            (List<Track> tracks, YoloDetection detection) = DeepSortMatcher.Run(framesPath + videoName + $"{i}.jpeg");

            watch.Stop();


            detections.Add( detection );
            yoloTime += watch.ElapsedMilliseconds;

            watch.Restart();

            //Recognize numbers with Tesseract
            TesseractPrediction[][] numbers = TesseractNumberRecognizer.Recognize(detection);

            watch.Stop();

            HashSet<string> currentFrame = [];

            //Check if a similar number is in the previous frame
            foreach (var item in numbers)
            {
                foreach(var prediction in item)
                {
                    if (prediction.Number != "NO")
                    {
                        foreach (var item1 in prevFrame)
                        {
                            if(item1.Contains(prediction.Number) || prediction.Number.Contains(item1))
                            {
                                currentFrame.Add(item1);
                            }
                        }
                        currentFrame.Add(prediction.Number);
                    }
                }
            }

            prevFrame = currentFrame;

            tesseractTime += watch.ElapsedMilliseconds;

            watch.Restart();

            Console.WriteLine("----------------------------------------------");
            Console.WriteLine("Lengths:");
            Console.WriteLine($"Detections: {detection.Detections.Count}");
            Console.WriteLine($"Tracks: {tracks.Count}");
            Console.WriteLine($"Numbers: {numbers.Length}");
            Console.WriteLine("----------------------------------------------");

            //Draw the results on the image
            for (int j = 0; j < numbers.Length; j++)
            {
                if (numbers[j].Length != 0)
                {
                    ObjectDetection det = detection.Detections[j];

                    TesseractPrediction maxPred = numbers[j].MaxBy(x => x.Confidence)!;

                    Cv2.Rectangle(detection.Image, YoloDetection.GetBounds(det), Scalar.Green, 2);

                    //Draw number on the car
                    Cv2.PutText(detection.Image, maxPred.Number, new Point(det.BoundingBox.Left + (det.BoundingBox.Width /3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.Black, 6);
                    Cv2.PutText(detection.Image, maxPred.Number, new Point(det.BoundingBox.Left + (det.BoundingBox.Width / 3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.White, 2);
                    
                    //Draw number list on the bottom of the image
                    Cv2.PutText(detection.Image, String.Join(",", currentFrame), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.Black, 10);
                    Cv2.PutText(detection.Image, String.Join(",", currentFrame), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                }

                
            }

            //Draw deepsort tracks
            for(int j =0; j < tracks.Count; j++)
            {
                tracks[j].Draw(detection.Image);
            }


            images.Add(detection.Image);
            watch.Stop();

            drawTime += watch.ElapsedMilliseconds;


        }

        watch.Restart();

        int width = images[0].Width;
        int height = images[0].Height;

        int framerate = 25;

        Console.WriteLine("writing video");

        VideoWriter writer = new(framesPath + "video_scritto.avi", FourCC.DIVX, framerate, new Size(width, height));

        foreach (Mat image in images)
        {
            writer.Write(image);
        }

        writer.Release();

        watch.Stop();

        //Print statistics

        long videoTime = watch.ElapsedMilliseconds;

        double fileSec = fileReadTime / 1000.0;
        double yoloSec = yoloTime / 1000.0;
        double tesseractSec = tesseractTime / 1000.0;
        double videoSec = videoTime / 1000.0;

        Console.WriteLine($"Retrieve({fileSec:F2}s) + Yolo({yoloSec:F2}s) + Tesseract({tesseractSec:F2}s) + Video({videoSec:F2}s) = {(fileSec + yoloSec + tesseractSec + videoSec):F2}s");

        Console.WriteLine($"FPS: {detections.Count} / ({fileSec:F2})s ({yoloSec:F2}s + {tesseractSec:F2}s)  = { (detections.Count / (fileSec + yoloSec + tesseractSec)):F2}");

        Console.WriteLine($"Avg retrieve: {(fileReadTime / detections.Count)}ms, Avg Yolo: {(yoloTime / detections.Count)}ms, Avg Tesseract: {(tesseractTime / detections.Count)}ms");

        TesseractNumberRecognizer.engine.Dispose();

        DeepSortMatcher.Dispose();

    }


}