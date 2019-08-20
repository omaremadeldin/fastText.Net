using System;
using System.Collections.Generic;
using System.IO;

namespace FastText
{
    internal class Program
    {
        static void PrintUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext <command> <args>\n\n" +
                "The commands supported by fasttext are:\n\n" +
                "  supervised              train a supervised classifier\n" +
                "  quantize                quantize a model to reduce the memory usage\n" +
                "  test                    evaluate a supervised classifier\n" +
                "  test-label              print labels with precision and recall scores\n" +
                "  predict                 predict most likely labels\n" +
                "  predict-prob            predict most likely labels with probabilities\n" +
                "  skipgram                train a skipgram model\n" +
                "  cbow                    train a cbow model\n" +
                "  print-word-vectors      print word vectors given a trained model\n" +
                "  print-sentence-vectors  print sentence vectors given a trained model\n" +
                "  print-ngrams            print ngrams given a trained model and word\n" +
                "  nn                      query for nearest neighbors\n" +
                "  analogies               query for analogies\n" +
                "  dump                    dump arguments,dictionary,input/output vectors\n");
        }

        static void PrintQuantizeUsage()
        {
            Console.Error.WriteLine("usage: fasttext quantize <args>");
        }

        static void PrintTestUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext test <model> <test-data> [<k>] [<th>]\n\n" +
                "  <model>      model filename\n" +
                "  <test-data>  test data filename (if -, read from stdin)\n" +
                "  <k>          (optional; 1 by default) predict top k labels\n" +
                "  <th>         (optional; 0.0 by default) probability threshold\n");
        }

        static void PrintPredictUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext predict[-prob] <model> <test-data> [<k>] [<th>]\n\n" +
                "  <model>      model filename\n" +
                "  <test-data>  test data filename (if -, read from stdin)\n" +
                "  <k>          (optional; 1 by default) predict top k labels\n" +
                "  <th>         (optional; 0.0 by default) probability threshold\n");
        }

        static void PrintTestLabelUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext test-label <model> <test-data> [<k>] [<th>]\n\n" +
                "  <model>      model filename\n" +
                "  <test-data>  test data filename\n" +
                "  <k>          (optional; 1 by default) predict top k labels\n" +
                "  <th>         (optional; 0.0 by default) probability threshold\n");
        }

        static void PrintPrintWordVectorsUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext print-word-vectors <model>\n\n" +
                "  <model>      model filename\n");
        }

        static void PrintPrintSentenceVectorsUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext print-sentence-vectors <model>\n\n" +
                "  <model>      model filename\n");
        }

        static void PrintPrintNgramsUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext print-ngrams <model> <word>\n\n" +
                "  <model>      model filename\n" +
                "  <word>       word to print\n");
        }

        static void PrintNNUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext nn <model> <k>\n\n" +
                "  <model>      model filename\n" +
                "  <k>          (optional; 10 by default) predict top k labels\n");
        }

        static void PrintAnalogiesUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext analogies <model> <k>\n\n" +
                "  <model>      model filename\n" +
                "  <k>          (optional; 10 by default) predict top k labels\n");
        }

        static void PrintDumpUsage()
        {
            Console.Error.WriteLine(
                "usage: fasttext dump <model> <option>\n\n" +
                "  <model>      model filename\n" +
                "  <option>     option from args,dict,input,output");
        }

        static void Quantize(string[] args)
        {
            var a = new Args();

            if (args.Length < 3)
            {
                PrintQuantizeUsage();
                a.PrintHelp();
                Environment.Exit(-1);
            }

            a.ParseArgs(args);
            var fasttext = new FastText();
            // parseArgs checks if a->output is given.
            fasttext.LoadModel(a.output + ".bin");
            fasttext.Quantize(a);
            fasttext.SaveModel(a.output + ".ftz");
        }

        static void Test(string [] args)
        {
            var perLabel = args[0] == "test-label";

            if (args.Length < 3 || args.Length > 5)
            {
                if (perLabel)
                {
                    PrintTestLabelUsage();
                }
                else
                {
                    PrintTestUsage();
                }

                Environment.Exit(-1);
            }

            var model = args[1];
            var input = args[2];
            int k = args.Length > 3 ? int.Parse(args[3]) : 1;
            var threshold = args.Length > 4 ? float.Parse(args[4]) : 0f;

            var fasttext = new FastText();
            fasttext.LoadModel(model);

            var meter = new Meter();

            if (input == "-")
            {
                fasttext.Test(Console.OpenStandardInput(), k, threshold, meter);
            }
            else
            {
                var ifs = new FileStream(input, FileMode.Open, FileAccess.Read);

                if (!ifs.CanRead)
                {
                    Console.Error.WriteLine("Test file cannot be opened!");
                    Environment.Exit(-1);
                }

                fasttext.Test(ifs, k, threshold, meter);
            }

            if (perLabel)
            {
                void writeMetric(string name, double value) 
                {
                    Console.Write($"{name} : ");
                    if (double.IsFinite(value))
                    {
                        Console.Write($"{value:0.######}");
                    }
                    else
                    {
                        Console.Write("--------");
                    }
                    Console.Write("  ");
                };

                var dict = fasttext.GetDictionary();
                for (int labelId = 0; labelId < dict.nlabels; labelId++)
                {
                    writeMetric("F1-Score", meter.F1Score(labelId));
                    writeMetric("Precision", meter.Precision(labelId));
                    writeMetric("Recall", meter.Recall(labelId));
                    Console.WriteLine($" {dict.GetLabel(labelId)}");
                }
            }
            meter.WriteGeneralMetrics(Console.Out, k);
        }

        static void PrintPredictions(
            List<Tuple<float, string>> predictions,
            bool printProb,
            bool multiline)
        {
            var first = true;
            foreach (var prediction in predictions)
            {
                if (!first && !multiline)
                {
                    Console.Write(" ");
                }

                first = false;
                Console.Write(prediction.Item2);

                if (printProb)
                {
                    Console.WriteLine($" {prediction.Item1}");
                }

                if (multiline)
                {
                    Console.Write(Environment.NewLine);
                }
            }

            if (!multiline)
            {
                Console.Write(Environment.NewLine);
            }
        }

        static void Predict(string[] args)
        {
            if (args.Length < 3 || args.Length > 5)
            {
                PrintPredictUsage();
                Environment.Exit(-1);
            }

            var k = 1;
            var threshold = 0f;

            if (args.Length > 3)
            {
                k = int.Parse(args[3]);
                if (args.Length == 5)
                {
                    threshold = float.Parse(args[4]);
                }
            }

            var printProb = args[0] == "predict-prob";
            var fasttext = new FastText();
            fasttext.LoadModel(args[1]);

            Stream ifs;

            var infile = args[2];
            var inputIsStdIn = infile == "-";

            if (!inputIsStdIn)
            {
                var fs = new FileStream(infile, FileMode.Open, FileAccess.Read);
                if (!inputIsStdIn && !fs.CanRead)
                {
                    Console.Error.WriteLine("Input file cannot be opened!");
                    Environment.Exit(-1);
                }

                ifs = fs;
            }
            else
            {
                ifs = Console.OpenStandardInput();
            }

            var predictions = new List<Tuple<float, string>>();

            while (fasttext.PredictLine(ifs, predictions, k, threshold))
            {
                PrintPredictions(predictions, printProb, false);
            }

            ifs.Close();
        }

        static void PrintWordVectors(string[] args)
        {
            if (args.Length != 2)
            {
                PrintPrintWordVectorsUsage();
                Environment.Exit(-1);
            }

            var fasttext = new FastText();
            fasttext.LoadModel(args[1]);

            var vec = new Vector(fasttext.GetDimension());

            while (Console.In.Peek() != -1)
            {
                var word = Console.ReadLine();

                fasttext.GetWordVector(vec, word);
                Console.WriteLine($"{word} {vec}");
            }
        }

        static void PrintSentenceVectors(string[] args)
        {
            if (args.Length != 2)
            {
                PrintPrintSentenceVectorsUsage();
                Environment.Exit(-1);
            }

            var fasttext = new FastText();
            fasttext.LoadModel(args[1]);

            var svec = new Vector(fasttext.GetDimension());

            while (Console.In.Peek() != -1)
            {
                fasttext.GetSentenceVector(Console.OpenStandardInput(), svec);
                // Don't print sentence
                Console.WriteLine(svec);
            }
        }

        static void PrintNgrams(string[] args)
        {
            if (args.Length != 3)
            {
                PrintPrintNgramsUsage();
                Environment.Exit(-1);
            }

            var fasttext = new FastText();
            fasttext.LoadModel(args[1]);

            var word = args[2];
            var ngramVectors = fasttext.GetNgramVectors(word);

            foreach (var ngramVector in ngramVectors)
            {
                Console.WriteLine($"{ngramVector.Item1} {ngramVector.Item2}");
            }
        }

        static void NN(string[] args)
        {
            var k = 0;

            if (args.Length == 2)
            {
                k = 10;
            }
            else if (args.Length == 3)
            {
                k = int.Parse(args[2]);
            }
            else
            {
                PrintNNUsage();
                Environment.Exit(-1);
            }

            var fasttext = new FastText();
            fasttext.LoadModel(args[1]);

            const string prompt = "Query word? ";
            Console.Write(prompt);

            while (Console.In.Peek() != -1)
            {
                var queryWord = Console.ReadLine();
                PrintPredictions(fasttext.GetNN(queryWord, k), true, true);
                Console.Write(prompt);
            }
        }

        static void Analogies(string[] args)
        {
            int k = 0;
            if (args.Length == 2)
            {
                k = 10;
            }
            else if (args.Length == 3)
            {
                k = int.Parse(args[2]);
            }
            else
            {
                PrintAnalogiesUsage();
                Environment.Exit(-1);
            }

            if (k <= 0)
            {
                throw new ArgumentException("k needs to be 1 or higher!");
            }

            var fasttext = new FastText();
            var model = args[1];
            Console.WriteLine($"Loading model {model}");
            fasttext.LoadModel(model);

            const string prompt = "Query triplet (A - B + C)? ";
            string wordA, wordB, wordC;

            Console.Write(prompt);

            while (true)
            {
                var words = Console.ReadLine().Split(' ');
                wordA = words[0];
                wordB = words[1];
                wordC = words[2];

                PrintPredictions(fasttext.GetAnalogies(k, wordA, wordB, wordC), true, true);

                Console.Write(prompt);
            }
        }

        static void train(string[] args)
        {
            var a = new Args();
            a.ParseArgs(args);

            var fasttext = new FastText();
            var outputFileName = a.output +".bin";

            var ofs = new FileStream(outputFileName, FileMode.CreateNew, FileAccess.Write);
            if (!ofs.CanWrite)
            {
                throw new ArgumentException($"{outputFileName} cannot be opened for saving.");
            }
            ofs.Close();

            fasttext.Train(a);
            fasttext.SaveModel(outputFileName);
            fasttext.SaveVectors(a.output + ".vec");

            if (a.saveOutput)
            {
                fasttext.SaveOutput(a.output + ".output");
            }
        }

        static void Dump(string[] args)
        {
            if (args.Length < 3)
            {
                PrintDumpUsage();
                Environment.Exit(-1);
            }

            var modelPath = args[1];
            var option = args[2];

            var fasttext = new FastText();
            fasttext.LoadModel(modelPath);

            if (option == "args")
            {
                fasttext.GetArgs().Dump(Console.Out);
            }
            else if (option == "dict")
            {
                fasttext.GetDictionary().Dump(Console.Out);
            }
            else if (option == "input")
            {
                if (fasttext.IsQuant())
                {
                    Console.Error.WriteLine("Not supported for quantized models.");
                }
                else
                {
                    fasttext.GetInputMatrix().Dump(Console.Out);
                }
            }
            else if (option == "output")
            {
                if (fasttext.IsQuant())
                {
                    Console.Error.WriteLine("Not supported for quantized models.");
                }
                else
                {
                    fasttext.GetOutputMatrix().Dump(Console.Out);
                }
            }
            else
            {
                PrintDumpUsage();
                Environment.Exit(-1);
            }
        }

        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                PrintUsage();
                return -1;
            }

            var command = args[0];
            if (command == "skipgram" || command == "cbow" || command == "supervised")
            {
                train(args);
            }
            else if (command == "test" || command == "test-label")
            {
                Test(args);
            }
            else if (command == "quantize")
            {
                Quantize(args);
            }
            else if (command == "print-word-vectors")
            {
                PrintWordVectors(args);
            }
            else if (command == "print-sentence-vectors")
            {
                PrintSentenceVectors(args);
            }
            else if (command == "print-ngrams")
            {
                PrintNgrams(args);
            }
            else if (command == "nn")
            {
                NN(args);
            }
            else if (command == "analogies")
            {
                Analogies(args);
            }
            else if (command == "predict" || command == "predict-prob")
            {
                Predict(args);
            }
            else if (command == "dump")
            {
                Dump(args);
            }
            else
            {
                PrintUsage();
                return -1;
            }

            return 0;
        }
    }
}