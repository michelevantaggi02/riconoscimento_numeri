using OpenCvSharp;
using riconoscimento_numeri.classes;
using riconoscimento_numeri.classes.DeepSort;
using SkiaSharp;
using Tesseract;
using YoloDotNet.Models;


class Program {



    const string IMG_PATH = @"imgs\";


    

    static void Main(string[] args) {

        RiconoscimentoYolo riconoscimento = new();
        string PROJECT_DIR = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        string path = Path.Combine(PROJECT_DIR, IMG_PATH, @"test_audi.png");

        Matcher DeepSortMatcher = new(@"models\yolov8m.onnx", @"models\FastReidVeRi.onnx");


        var watch = System.Diagnostics.Stopwatch.StartNew();

        

        List<YoloDetection> detections = new List<YoloDetection>();

        int frames = 234;

        long fileReadTime = 0;
        long yoloTime = 0;
        long tesseractTime = 0;
        long drawTime = 0;

        List<Mat> images = [];

        HashSet<string> prevFrame = [];

        for (int i = 1; i <= frames; i++)
        {
            watch.Restart();
            (List<Track> tracks, YoloDetection detection) = DeepSortMatcher.Run(@"C:\Users\michi\Desktop\riconoscimento_numeri\vids\corti\frames\video1\" + $"{i}.jpeg");

            watch.Stop();


            detections.Add( detection );
            yoloTime += watch.ElapsedMilliseconds;

            watch.Restart();

            TesseractPrediction[][] numbers = RiconoscimentoTesseract.Recognize(detection);

            watch.Stop();

            HashSet<string> currentFrame = [];

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

            for (int j = 0; j < numbers.Length; j++)
            {
                if (numbers[j].Length != 0)
                {
                    ObjectDetection det = detection.Detections[j];

                    TesseractPrediction maxPred = numbers[j].MaxBy(x => x.Confidence)!;
                    Cv2.PutText(detection.Image, maxPred.Number, new Point(det.BoundingBox.Left + (det.BoundingBox.Width /3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.Black, 6);
                    Cv2.PutText(detection.Image, maxPred.Number, new Point(det.BoundingBox.Left + (det.BoundingBox.Width / 3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.White, 2);
                    Cv2.PutText(detection.Image, String.Join(",", currentFrame), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.Black, 10);
                    Cv2.PutText(detection.Image, String.Join(",", currentFrame), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                }

                
            }

            for(int j =0; j < tracks.Count; j++)
            {
                Track track = tracks[j];
                Cv2.PutText(detection.Image, $"ID: {track.id}", new Point(track.currentBounds.Left + (track.currentBounds.Width), track.currentBounds.Top + (track.currentBounds.Height)), HersheyFonts.HersheySimplex, 2, Scalar.Black, 6);
                Cv2.PutText(detection.Image, $"ID: {track.id}", new Point(track.currentBounds.Left + (track.currentBounds.Width), track.currentBounds.Top + (track.currentBounds.Height)), HersheyFonts.HersheySimplex, 2, Scalar.White, 2);
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

        VideoWriter writer = new("C:\\Users\\michi\\Desktop\\riconoscimento_numeri\\vids\\corti\\video1_scritto.avi", FourCC.DIVX, framerate, new Size(width, height));

        foreach (Mat image in images)
        {
            writer.Write(image);
        }

        writer.Release();

        watch.Stop();

        long videoTime = watch.ElapsedMilliseconds;

        double fileSec = fileReadTime / 1000.0;
        double yoloSec = yoloTime / 1000.0;
        double tesseractSec = tesseractTime / 1000.0;
        double videoSec = videoTime / 1000.0;

        Console.WriteLine($"Retrieve({fileSec:F2}s) + Yolo({yoloSec:F2}s) + Tesseract({tesseractSec:F2}s) + Video({videoSec:F2}s) = {(fileSec + yoloSec + tesseractSec + videoSec):F2}s");

        Console.WriteLine($"FPS: {detections.Count} / ({fileSec:F2})s ({yoloSec:F2}s + {tesseractSec:F2}s)  = { (detections.Count / (fileSec + yoloSec + tesseractSec)):F2}");

        Console.WriteLine($"Avg retrieve: {(fileReadTime / detections.Count)}ms, Avg Yolo: {(yoloTime / detections.Count)}ms, Avg Tesseract: {(tesseractTime / detections.Count)}ms");
        riconoscimento.model.Dispose();
        RiconoscimentoTesseract.engine.Dispose();


        //test tesseract
        /*using TesseractEngine engine = new(@"models", "ita", EngineMode.Default);

        //engine.SetVariable("tessedit_char_whitelist", "0123456789");
        //engine.SetVariable("outputbase", "digits");

        using Pix pixImage = Pix.LoadFromFile(Path.Combine(IMG_PATH, "test_numero_largo.PNG"));


        using var page = engine.Process(pixImage);

        string text = page.GetText();

        Console.WriteLine($"{text}");*/

    }


}