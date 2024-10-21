// See https://aka.ms/new-console-template for more information
using Google.Apis.Auth.OAuth2;
using Google.Cloud.Storage.V1;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;

internal class Program
{
    private static async Task Main(string[] args)
    {
        DownloadFile();

        // Load and run the ONNX model using ONNX Runtime
        InferenceSession session = new InferenceSession("model.onnx");

        String img_filename = "C:\\Users\\jens.baetens3\\OneDrive - ODISEE\\Lesmateriaal\\MachineLearning\\Lessen\\03_Computervisie\\object-detection\\img1.jpg";
        DenseTensor<float> image = LoadImageAsTensor(img_filename, 224, 224);

        Tensor<float> outputTensor = Predict(session, image);

        int prediction = GetPredictedClass(outputTensor);
        Console.WriteLine($"Most likely class: {prediction}");

        Console.WriteLine("Done");
    }

    static void DownloadFile()
    {
        GoogleCredential credential = GoogleCredential.FromFile("C:\\Users\\jens.baetens3\\OneDrive - ODISEE\\Lesmateriaal\\MachineLearning\\Lessen\\03_Computervisie\\firebase_config_server.json");
        StorageClient storageClient = StorageClient.Create(credential);

        // eg: Images/MyImage.jpg
        string objectName = "models/07_model.onnx";

        // eg: "c://CallItSomethingElseIfYouWant.jpg"
        string localPath = "model.onnx";

        // eg: xxxxxxxx-xxxxx.appspot.com
        string bucketName = "ml-deployment-707dd.appspot.com";

        FileStream fileStream = File.OpenWrite(@localPath);

        // --- The Magic Line
        storageClient.DownloadObject(bucketName, objectName, fileStream);
        // --  

        fileStream.Flush();
        fileStream.Close();
    }

    static DenseTensor<float> LoadImageAsTensor(string imagePath, int width, int height)
    {
        using var bitmap = new Bitmap(imagePath);
        var resized = new Bitmap(bitmap, new Size(width, height));

        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = resized.GetPixel(x, y);
                tensor[0, 0, y, x] = pixel.R / 255.0f; // Red channel
                tensor[0, 1, y, x] = pixel.G / 255.0f; // Green channel
                tensor[0, 2, y, x] = pixel.B / 255.0f; // Blue channel
            }
        }

        return tensor;
    }

    static Tensor<float> Predict(InferenceSession session, DenseTensor<float> input)
    {
        // Create input tensor
        var inputName = session.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, input) };

        // Run the model
        using var results = session.Run(inputs);

        // Process the results
        var outputTensor = results.First().AsTensor<float>();

        return outputTensor;
    }
    static int GetPredictedClass(Tensor<float> outputTensor)
    {
        // Assuming the model outputs a tensor of probabilities
        var probabilities = outputTensor.ToArray();
        var maxIndex = Array.IndexOf(probabilities, probabilities.Max());
        return maxIndex;
    }
}