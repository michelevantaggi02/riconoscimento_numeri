
using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace riconoscimento_numeri.classes.DeepSort
{

    internal class FastReID
    {
        private InferenceSession session;
        private string inputName = "batched_inputs.1";
        private string outputName = "924";
        private int[] inputShape;
        private int[] outputShape;
        private int inputBufferSize;
        private int outputBufferSize;
        private ArrayPool<float> pool;



        public FastReID(string modelPath)
        {
            session = new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider());

            inputName = session.InputNames[0];
            outputName = session.OutputNames[0];

            inputShape = session.InputMetadata[inputName].Dimensions;
            outputShape = session.OutputMetadata[outputName].Dimensions;

            inputBufferSize = 1;
            outputBufferSize = 1;

            for (int i = 0; i < inputShape.Length; i++)
            {
                inputBufferSize *= inputShape[i];
            }

            for (int i = 0; i < outputShape.Length; i++)
            {
                outputBufferSize *= outputShape[i];
            }

            pool = ArrayPool<float>.Create(maxArrayLength: inputBufferSize + 1, maxArraysPerBucket: 10);
        }

        public void PrintInputInfo()
        {
            foreach (var inputMeta in session.InputMetadata)
            {
                var value = inputMeta.Value;
                Console.WriteLine($"Input Name: {inputMeta.Key}");
                Console.WriteLine($"Input Dimensions: {string.Join(",", value.Dimensions)}");
                Console.WriteLine($"Input Element Type: {value.ElementType}");
            }
        }

        public void PrintOutputInfo()
        {
            foreach (var outputMeta in session.OutputMetadata)
            {
                var value = outputMeta.Value;
                Console.WriteLine($"Output Name: {outputMeta.Key}");
                Console.WriteLine($"Output Dimensions: {string.Join(",", value.Dimensions)}");
                Console.WriteLine($"Output Element Type: {value.ElementType}");
            }
        }


        public Detail Recognize(Mat image)
        {
            float[] buffer = pool.Rent(minimumLength: inputBufferSize);

            Mat rgb = new();
            Cv2.CvtColor(image, rgb, ColorConversionCodes.BGR2RGB);

            try
            {
                DenseTensor<float> inputTensor = Prepare(rgb, buffer);

                long[] longInputShape = new long[inputShape.Length];

                for (int i = 0; i < inputShape.Length; i++)
                {
                    longInputShape[i] = inputShape[i];
                }

                using OrtValue input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputTensor.Buffer, longInputShape);

                Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>()
                {
                    { inputName, input }
                };

                IDisposableReadOnlyCollection<OrtValue> recognition = session.Run(new RunOptions(), inputs, [outputName]);
                Console.WriteLine(recognition.Count);

                return new Detail(recognition[0].GetTensorDataAsSpan<float>().ToArray());

            }
            finally
            {
                pool.Return(buffer, true);


            }
        }
        //https://github.com/NickSwardh/YoloDotNet/blob/master/YoloDotNet/Extensions/ImageExtension.cs#L134
        unsafe private DenseTensor<float> Prepare(Mat image, float[] buffer)
        {


            Mat resized = image.Resize(new Size(inputShape[2], inputShape[3]));

            byte* pixels = resized.DataPointer;

            int pixelIndex = 0;
            int pixelsPerChannel = inputBufferSize / 3;
            int offset = 0;

            for (int y = 0; y < inputShape[2]; y++)
            {
                for (int x = 0; x < inputShape[3]; x++, pixelIndex++, offset += 3)
                {
                    offset = (y * inputShape[3] + x) * 3;
                    var r = pixels[offset];
                    var g = pixels[offset + 1];
                    var b = pixels[offset + 2];

                    if ((r | g | b) == 0)
                        continue;

                    buffer[pixelIndex] = r / 255.0f;
                    buffer[pixelsPerChannel + pixelIndex] = g / 255.0f;
                    buffer[pixelsPerChannel * 2 + pixelIndex] = b / 255.0f;
                }
            }

            return new DenseTensor<float>(buffer.AsMemory()[..inputBufferSize], inputShape);

        }
    }
}
