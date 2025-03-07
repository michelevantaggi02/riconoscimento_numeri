using OpenCvSharp;
using riconoscimento_numeri.classes;
using SkiaSharp;
using Tesseract;
using YoloDotNet.Models;


class Program {



    const string IMG_PATH = @"imgs\";


    

    static void Main(string[] args) {

        RiconoscimentoYolo riconoscimento = new();
        string PROJECT_DIR = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        string path = Path.Combine(PROJECT_DIR, IMG_PATH, @"test_audi.png");

        //riconoscimento.recognize(path);

        var watch = System.Diagnostics.Stopwatch.StartNew();

        /*YoloDetection[] detections = riconoscimento.recognize_video(@"C:\Users\michi\Desktop\riconoscimento_numeri\vids\corti\video1.mp4");

        watch.Stop();

        long yoloTime = watch.ElapsedMilliseconds;

        watch.Restart();

        foreach (YoloDetection detection in detections)
        {
            int[] numbers = RiconoscimentoTesseract.Recognize(detection);

            for (int i = 0; i < numbers.Length; i++)
            {
                if (numbers[i] == -1)
                {
                    continue;
                }

                ObjectDetection det = detection.Detections[i];
                Cv2.PutText(detection.Image, numbers[i].ToString(), new Point(det.BoundingBox.Left, det.BoundingBox.Top), HersheyFonts.HersheySimplex, 1, Scalar.Black, 4);
                Cv2.PutText(detection.Image, numbers[i].ToString(), new Point(det.BoundingBox.Left, det.BoundingBox.Top), HersheyFonts.HersheySimplex, 1, Scalar.White, 3);
            }


            Console.WriteLine("Detected numbers: ");
            Console.WriteLine(String.Join(", ", numbers));

            Cv2.ImShow("test", detection.Image);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        watch.Stop();
        long tesseractTime = watch.ElapsedMilliseconds;

        watch.Restart();

        Mat[] images = detections.Select(detections => detections.Image).ToArray();*/

        List<YoloDetection> detections = new List<YoloDetection>();

        int frames = 234;

        long fileReadTime = 0;
        long yoloTime = 0;
        long tesseractTime = 0;
        long drawTime = 0;

        List<Mat> images = [];

        HashSet<int> passati = [];

        for (int i = 1; i <= frames; i++)
        {
            watch.Restart();
            using var image = SKImage.FromEncodedData(@"C:\Users\michi\Desktop\riconoscimento_numeri\vids\corti\frames\video1\" + $"{i}.jpeg");
            watch.Stop();

            fileReadTime += watch.ElapsedMilliseconds;

            watch.Restart();

            var result = riconoscimento.model.RunObjectDetection(image);

            using SKBitmap bitmap = SKBitmap.FromImage(image);

            SKPixmap pixmap = new();

            if (!bitmap.PeekPixels(pixmap))
            {
                throw new Exception("Impossibile ottenere i pixel");
            }

            IntPtr data = pixmap.GetPixels();

            Mat mat = Mat.FromPixelData(bitmap.Height, bitmap.Width, MatType.CV_8UC4, data);

            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);
            YoloDetection detection = new YoloDetection
            {
                Detections = result.Where(x => x.Label.Name.Equals("car")).ToList(),
                Image = mat,
            };

            watch.Stop();

            detections.Add( detection );
            yoloTime += watch.ElapsedMilliseconds;

            watch.Restart();

            int[] numbers = RiconoscimentoTesseract.Recognize(detection);

            watch.Stop();

            foreach (var item in numbers)
            {
                passati.Add(item);
            }

            tesseractTime += watch.ElapsedMilliseconds;

            watch.Restart();

            for (int j = 0; j < numbers.Length; j++)
            {
                if (numbers[j] != -1)
                {
                    ObjectDetection det = detection.Detections[j];
                    Cv2.PutText(detection.Image, numbers[j].ToString(), new Point(det.BoundingBox.Left + (det.BoundingBox.Width /3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.Black, 6);
                    Cv2.PutText(detection.Image, numbers[j].ToString(), new Point(det.BoundingBox.Left + (det.BoundingBox.Width / 3), det.BoundingBox.Top + (det.BoundingBox.Height / 3)), HersheyFonts.HersheySimplex, 2, Scalar.White, 2);
                    Cv2.PutText(detection.Image, String.Join(",", passati), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.Black, 10);
                    Cv2.PutText(detection.Image, String.Join(",", passati), new Point(5, detection.Image.Height - 10), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                }

                
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