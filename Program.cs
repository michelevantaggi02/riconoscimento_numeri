

using SkiaSharp;
using Tesseract;
using YoloDotNet;
using YoloDotNet.Extensions;
using YoloDotNet.Models;



class Program {


    const string IMG_PATH = @"D:\riconoscimento_numeri\imgs\";

    static void Main(string[] args) {

        var model = new Yolo(new YoloOptions {
            OnnxModel = @"models\yolov8m.onnx",
            ModelType = YoloDotNet.Enums.ModelType.ObjectDetection,
            Cuda = false,
        });

        using var image = SKImage.FromEncodedData(IMG_PATH + @"img1.jpg");

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

        resultImage.Save(IMG_PATH + @"new_image.jpg", SKEncodedImageFormat.Jpeg, 80);

        using var engine = new TesseractEngine(@"models", "ita", EngineMode.Default);


        for (int i = 0; i < result.Count; i++) {
            var item = result[i];
            using var temp = image.Subset(item.BoundingBox);

            string temp_path = IMG_PATH + @"cuts\" + i + ".jpg";
            temp.Save(temp_path, SKEncodedImageFormat.Jpeg, 80);


            using Pix pixImage = Pix.LoadFromFile(temp_path);

            using var page = engine.Process(pixImage);

            string text = page.GetText();

            Console.WriteLine($"Item {i}: {text}");
            

        }



    }

    
}

