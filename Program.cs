

using SkiaSharp;
using Tesseract;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

class Program {
    static void Main(string[] args) {

        var model = new Yolo(new YoloOptions {
            OnnxModel = @"models/yolov8m.onnx",
            ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
            Cuda = false,
        });

        using var image = SKImage.FromEncodedData(@"imgs/img1.jpg");

        var result = model.RunObjectDetection(image);

        List<string> labels = [];

        foreach (var item in result) {
            if (!labels.Contains(item.Label.Name)) {
                labels.Add(item.Label.Name);
            }

        }

        labels.ForEach(x => Console.WriteLine(x));

        result = result.Where(x => x.Confidence > 0.7 && x.Label.Name.Equals("car")).ToList();

        using var resultImage = image.Draw(result);

        resultImage.Save(@"imgs/new_image.jpg", SKEncodedImageFormat.Jpeg, 80);

        var engine = new TesseractEngine(@"models/ita.traineddata", "ita", EngineMode.Default) {
            DefaultPageSegMode = PageSegMode.SingleWord
        };


        for (int i = 0; i < result.Count; i++) {
            var item = result[i];
            using var temp = image.Subset(item.BoundingBox);

            using var bitmap = SKBitmap.FromImage(temp);

            using var page = engine.Process(Pix.LoadFromMemory(bitmap.Bytes));

            string text = page.GetText();

            Console.WriteLine($"Item {i}: {text}");
            
            
            

            temp.Save(@"imgs/cuts/" + i + ".jpg", SKEncodedImageFormat.Jpeg, 80);
        }

    }

    
}

