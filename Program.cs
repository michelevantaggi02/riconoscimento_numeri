using riconoscimento_numeri.classes;
using Tesseract;


class Program {



    const string IMG_PATH = @"imgs\";


    

    static void Main(string[] args) {

        Riconoscimento riconoscimento = new();
        string PROJECT_DIR = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        string path = Path.Combine(PROJECT_DIR, IMG_PATH, @"test_audi.png");

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