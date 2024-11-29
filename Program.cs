using riconoscimento_numeri.classes;


class Program {


    const string IMG_PATH = @"D:\riconoscimento_numeri\imgs\";


    

    static void Main(string[] args) {

        Riconoscimento riconoscimento = new();

        string path = Path.Combine(IMG_PATH, @"mini\");

        riconoscimento.recognize(path);

    }

    
}