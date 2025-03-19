
namespace riconoscimento_numeri.classes.DeepSort
{
    public class Detail
    {
        public double[] values { get; init; }
        public double magnitude { get; private set; }

        public Detail(double[] values)
        {
            this.values = values;

            magnitude = GetMagnitude();
        }

        public Detail(float[] values)
        {
            this.values = Array.ConvertAll(values, x => (double)x);

            magnitude = GetMagnitude();
        }

        private double GetMagnitude()
        {
            double sum = 0;

            for (int i = 0; i < values.Length; i++)
            {
                sum += values[i] * values[i];
            }

            return Math.Sqrt(sum);
        }

        public void Normalize()
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = values[i] / magnitude;
            }
            magnitude = GetMagnitude();
        }

        public double CosineDistance(Detail other)
        {
            return 1 - (this * other / (this.magnitude * other.magnitude));
        }

        public static Detail operator +(Detail a, Detail b)
        {
            double[] result = new double[a.values.Length];
            for (int i = 0; i < a.values.Length; i++)
            {
                result[i] = a.values[i] + b.values[i];
            }
            return new Detail(result);
        }

        public static double operator *(Detail a, Detail b)
        {
            double result = 0;
            for (int i = 0; i < a.values.Length; i++)
            {
                result += a.values[i] * b.values[i];
            }
            return result;
        }

        public static Detail operator /(Detail a, double b)
        {
            double[] result = new double[a.values.Length];
            for (int i = 0; i < a.values.Length; i++)
            {
                result[i] = a.values[i] / b;
            }
            return new Detail(result);
        }

    }
}
