using riconoscimento_numeri.classes;
using Tesseract;


class Program {


    const string IMG_PATH = @"D:\riconoscimento_numeri\imgs\";


    

    static void Main(string[] args) {

        Riconoscimento riconoscimento = new();

        string path = Path.Combine(IMG_PATH, @"mini\");

        riconoscimento.recognize(path);

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