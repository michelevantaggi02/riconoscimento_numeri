
namespace riconoscimento_numeri.classes
{
    /// <summary>
    /// Tesseract prediction result class
    /// </summary>
    public class TesseractPrediction
    {
        /// <summary>
        /// Recognized number
        /// </summary>
        public string Number { get; set; }

        /// <summary>
        /// Confidence of the recognition
        /// </summary>
        public float Confidence { get; set; }
    }
}
