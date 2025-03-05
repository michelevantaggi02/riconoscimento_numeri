using OpenCvSharp;
using riconoscimento_numeri.classes;
using Tesseract;
using YoloDotNet.Models;


class Program {



    const string IMG_PATH = @"imgs\";


    

    static void Main(string[] args) {

        RiconoscimentoYolo riconoscimento = new();
        string PROJECT_DIR = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        string path = Path.Combine(PROJECT_DIR, IMG_PATH, @"test_audi.png");

        //riconoscimento.recognize(path);

        YoloDetection[] detections = riconoscimento.recognize_video(@"C:\Users\michi\Desktop\riconoscimento_numeri\vids\corti\video1.mp4");

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
                Cv2.PutText(detection.Image, numbers[i].ToString(), new Point(det.BoundingBox.Left, det.BoundingBox.Top), HersheyFonts.HersheySimplex, 2, Scalar.Black, 2);
                Cv2.PutText(detection.Image, numbers[i].ToString(), new Point(det.BoundingBox.Left, det.BoundingBox.Top), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
            }


            /*Console.WriteLine("Detected numbers: ");
            Console.WriteLine(String.Join(", ", numbers));

            Cv2.ImShow("test", detection.Image);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();*/
        }


        Mat[] images = detections.Select(detections => detections.Image).ToArray();

        int width = images[0].Width;
        int height = images[0].Height;

        int framerate = 25;

        Console.WriteLine("writing video");

        VideoWriter writer = new("C:\\Users\\michi\\Desktop\\riconoscimento_numeri\\vids\\corti\\video1_scritto.avi", FourCC.XVID, framerate, new Size(width, height));

        foreach (Mat image in images)
        {
            writer.Write(image);
        }

        writer.Release();
        riconoscimento.model.Dispose();


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