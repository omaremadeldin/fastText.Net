using System;
using System.Collections.Generic;
using System.IO;

namespace fasttext
{
    public enum model_name : int { cbow = 1, sg, sup };
    public enum loss_name : int { hs = 1, ns, softmax, ova };

    public class Args
    {   
        public string input;
        public string output;

        public double lr = 0.05;
        public int lrUpdateRate = 100;
        public int dim = 100;
        public int ws = 5;
        public int epoch = 5;
        public int minCount = 5;
        public int minCountLabel = 0;
        public int neg = 5;
        public int wordNgrams = 1;
        public loss_name loss = loss_name.ns;
        public model_name model = model_name.sg;
        public int bucket = 2000000;
        public int minn = 3;
        public int maxn = 6;
        public int thread = 12;
        public double t = 1E-4;
        public string label = "__label__";
        public int verbose = 2;
        public string pretrainedVectors = "";
        public bool saveOutput = false;

        public bool qout = false;
        public bool retrain = false;
        public bool qnorm = false;
        public int cutoff = 0;
        public int dsub = 2;

        protected string lossToString(loss_name ln)
        {
            switch (ln)
            {
                case loss_name.hs:
                    return "hs";
                case loss_name.ns:
                    return "ns";
                case loss_name.softmax:
                    return "softmax";
                case loss_name.ova:
                    return "one-vs-all";
                default:
                    return "Unknown loss!"; // should never happen
            }
        }

        protected string boolToString(bool b)
        {
            if (b)
            {
                return "true";
            }
            else
            {
                return "false";
            }
        }

        protected string modelToString(model_name mn)
        {
            switch (mn)
            {
                case model_name.cbow:
                    return "cbow";
                case model_name.sg:
                    return "sg";
                case model_name.sup:
                    return "sup";
                default:
                    return "Unknown model name!"; // should never happen
            }
        }

        public void parseArgs(IReadOnlyList<string> args)
        {
            var command = args[1];

            if (command == "supervised")
            {
                model = model_name.sup;
                loss = loss_name.softmax;
                minCount = 1;
                minn = 0;
                maxn = 0;
                lr = 0.1;
            }
            else if (command == "cbow")
            {
                model = model_name.cbow;
            }

            for (int ai = 2; ai < args.Count; ai += 2)
            {
                if (args[ai][0] != '-')
                {
                    Console.Error.WriteLine("Provided argument without a dash! Usage:");
                    printHelp();
                    Environment.Exit(-1);
                }
                try
                {
                    if (args[ai] == "-h")
                    {
                        Console.Error.WriteLine("Here is the help! Usage:");
                        printHelp();
                        Environment.Exit(-1);
                    }
                    else if (args[ai] == "-input")
                    {
                        input = args[ai + 1];
                    }
                    else if (args[ai] == "-output")
                    {
                        output = args[ai + 1];
                    }
                    else if (args[ai] == "-lr")
                    {
                        lr = float.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-lrUpdateRate")
                    {
                        lrUpdateRate = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-dim")
                    {
                        dim = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-ws")
                    {
                        ws = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-epoch")
                    {
                        epoch = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-minCount")
                    {
                        minCount = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-minCountLabel")
                    {
                        minCountLabel = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-neg")
                    {
                        neg = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-wordNgrams")
                    {
                        wordNgrams = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-loss")
                    {
                        if (args[ai + 1] == "hs")
                        {
                            loss = loss_name.hs;
                        }
                        else if (args[ai + 1] == "ns")
                        {
                            loss = loss_name.ns;
                        }
                        else if (args[ai + 1] == "softmax")
                        {
                            loss = loss_name.softmax;
                        }
                        else if (args[ai + 1] == "one-vs-all" || args[ai + 1] == "ova")
                        {
                            loss = loss_name.ova;
                        }
                        else
                        {
                            Console.Error.WriteLine($"Unknown loss: {args[ai + 1]}");
                            printHelp();
                            Environment.Exit(-1);
                        }
                    }
                    else if (args[ai] == "-bucket")
                    {
                        bucket = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-minn")
                    {
                        minn = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-maxn")
                    {
                        maxn = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-thread")
                    {
                        thread = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-t")
                    {
                        t = float.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-label")
                    {
                        label = args[ai + 1];
                    }
                    else if (args[ai] == "-verbose")
                    {
                        verbose = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-pretrainedVectors")
                    {
                        pretrainedVectors = args[ai + 1];
                    }
                    else if (args[ai] == "-saveOutput")
                    {
                        saveOutput = true;
                        ai--;
                    }
                    else if (args[ai] == "-qnorm")
                    {
                        qnorm = true;
                        ai--;
                    }
                    else if (args[ai] == "-retrain")
                    {
                        retrain = true;
                        ai--;
                    }
                    else if (args[ai] == "-qout")
                    {
                        qout = true;
                        ai--;
                    }
                    else if (args[ai] == "-cutoff")
                    {
                        cutoff = int.Parse(args[ai + 1]);
                    }
                    else if (args[ai] == "-dsub")
                    {
                        dsub = int.Parse(args[ai + 1]);
                    }
                    else
                    {
                        Console.Error.WriteLine($"Unknown argument: {args[ai]}");
                        printHelp();
                        Environment.Exit(-1);
                    }
                }
                catch (IndexOutOfRangeException)
                {
                    Console.Error.WriteLine($"{args[ai]} is missing an argument");
                    printHelp();
                    Environment.Exit(-1);
                }
            }

            if (string.IsNullOrEmpty(input) || string.IsNullOrEmpty(output))
            {
                Console.Error.WriteLine("Empty input or output path.");
                printHelp();
                Environment.Exit(-1);
            }

            if (wordNgrams <= 1 && maxn == 0)
            {
                bucket = 0;
            }
        }

        public void printHelp()
        {
            printBasicHelp();
            printDictionaryHelp();
            printTrainingHelp();
            printQuantizationHelp();
        }

        public void printBasicHelp()
        {
            Console.Error.Write(
                "\nThe following arguments are mandatory:\n" +
                "  -input              training file path\n" +
                "  -output             output file path\n" +
                "\nThe following arguments are optional:\n" +
                $"  -verbose            verbosity level [{verbose}]\n");
        }

        public void printDictionaryHelp()
        {
            Console.Error.Write(
                "\nThe following arguments for the dictionary are optional:\n" +
                $"  -minCount           minimal number of word occurences [{minCount}]\n" +
                $"  -minCountLabel      minimal number of label occurences [{minCountLabel}]\n" +
                $"  -wordNgrams         max length of word ngram [{wordNgrams}]\n" +
                $"  -bucket             number of buckets [{bucket}]\n" +
                $"  -minn               min length of char ngram [{minn}]\n" +
                $"  -maxn               max length of char ngram [{maxn}]\n" +
                $"  -t                  sampling threshold [{t}]\n" +
                $"  -label              labels prefix [{label}]\n");
        }

        public void printTrainingHelp()
        {
            Console.Error.Write(
                "\nThe following arguments for training are optional:\n" +
                $"  -lr                 learning rate [{lr}]\n" +
                $"  -lrUpdateRate       change the rate of updates for the learning rate [{lrUpdateRate}]\n" +
                $"  -dim                size of word vectors [{dim}]\n" +
                $"  -ws                 size of the context window [{ws}]\n" +
                $"  -epoch              number of epochs [{epoch}]\n" +
                $"  -neg                number of negatives sampled [{neg}]\n" +
                $"  -loss               loss function {{ns, hs, softmax, one-vs-all}} [{lossToString(loss)}]\n" +
                $"  -thread             number of threads [{thread}]\n" +
                $"  -pretrainedVectors  pretrained word vectors for supervised learning [{pretrainedVectors}]\n" +
                $"  -saveOutput         whether output params should be saved [{boolToString(saveOutput)}]\n");
        }

        public void printQuantizationHelp()
        {
            Console.Error.Write(
                "\nThe following arguments for quantization are optional:\n" +
                $"  -cutoff             number of words and ngrams to retain [{cutoff}]\n" +
                $"  -retrain            whether embeddings are finetuned if a cutoff is applied [{boolToString(retrain)}]\n" +
                $"  -qnorm              whether the norm is quantized separately [{boolToString(qnorm)}]\n" +
                $"  -qout               whether the classifier is quantized [{boolToString(qout)}]\n" +
                $"  -dsub               size of each sub-vector [{dsub}]\n");
        }

        public void save(BinaryWriter writer)
        {
            writer.Write(dim);
            writer.Write(ws);
            writer.Write(epoch);
            writer.Write(minCount);
            writer.Write(neg);
            writer.Write(wordNgrams);
            writer.Write((int)loss);
            writer.Write((int)model);
            writer.Write(bucket);
            writer.Write(minn);
            writer.Write(maxn);
            writer.Write(lrUpdateRate);
            writer.Write(t);
        }

        public void load(BinaryReader reader)
        {
            dim = reader.ReadInt32();
            ws = reader.ReadInt32();
            epoch = reader.ReadInt32();
            minCount = reader.ReadInt32();
            neg = reader.ReadInt32();
            wordNgrams = reader.ReadInt32();
            loss = (loss_name)reader.ReadInt32();
            model = (model_name)reader.ReadInt32();
            bucket = reader.ReadInt32();
            minn = reader.ReadInt32();
            maxn = reader.ReadInt32();
            lrUpdateRate = reader.ReadInt32();
            t = reader.ReadDouble();
        }

        public void dump(TextWriter writer)
        {
            writer.WriteLine($"dim {dim}");
            writer.WriteLine($"ws {ws}");
            writer.WriteLine($"epoch {epoch}");
            writer.WriteLine($"minCount {minCount}");
            writer.WriteLine($"neg {neg}");
            writer.WriteLine($"wordNgrams {wordNgrams}");
            writer.WriteLine($"loss {lossToString(loss)}");
            writer.WriteLine($"model {modelToString(model)}");
            writer.WriteLine($"bucket {bucket}");
            writer.WriteLine($"minn {minn}");
            writer.WriteLine($"maxn {maxn}");
            writer.WriteLine($"lrUpdateRate {lrUpdateRate}");
            writer.WriteLine($"t {t}");
        }
    };
}