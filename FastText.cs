using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Wintellect.PowerCollections;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

#pragma warning disable CS0618 // Type or member is obsolete

namespace FastText
{
    public class FastText
    {
        private const int FASTTEXT_VERSION = 12; /* Version 1b */
        private const int FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

        protected Args args_;
        protected Dictionary dict_;

        protected Matrix input_;
        protected Matrix output_;

        protected Model model_;

        protected long tokenCount_;
        protected float loss_;

        protected TimeSpan start_;

        protected bool quant_;
        protected int version;
        protected DenseMatrix wordVectors_;

        protected void SignModel(BinaryWriter writer)
        {
            writer.Write(FASTTEXT_FILEFORMAT_MAGIC_INT32);
            writer.Write(FASTTEXT_VERSION);
        }

        protected bool CheckModel(BinaryReader reader)
        {
            var magic = reader.ReadInt32();
            if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32)
            {
                return false;
            }

            var version = reader.ReadInt32();
            if (version > FASTTEXT_VERSION)
            {
                return false;
            }

            return true;
        }

        protected void StartThreads()
        {
            start_ = DateTime.Now.TimeOfDay;
            tokenCount_ = 0;
            loss_ = -1;

            var threads = new List<Thread>();
            for (int i = 0; i < args_.thread; i++)
            {
                threads.Add(new Thread(() => TrainThread(i)));
            }

            var ntokens = dict_.ntokens;
            // Same condition as trainThread
            while (tokenCount_ < args_.epoch * ntokens)
            {
                Thread.Sleep(100);
                if (loss_ >= 0 && args_.verbose > 1)
                {
                    var progress = (float)tokenCount_ / (args_.epoch * ntokens);
                    Console.Error.Write("\r");
                    PrintInfo(progress, loss_, Console.Error);
                }
            }

            for (int i = 0; i < args_.thread; i++)
            {
                threads[i].Join();
            }

            if (args_.verbose > 0)
            {
                Console.Error.Write("\r");
                PrintInfo(1f, loss_, Console.Error);
                Console.Error.Write(Environment.NewLine);
            }
        }

        protected void AddInputVector(Vector vec, int ind)
        {
            vec.AddRow(input_, ind);
        }

        protected void TrainThread(int threadId)
        {
            var ifs = new FileStream(args_.input, FileMode.Open, FileAccess.Read);
            ifs.Flush();
            ifs.Seek(threadId * ifs.Length / args_.thread, SeekOrigin.Begin);

            var state = new Model.State(args_.dim, (int)output_.Size(0), threadId);

            var ntokens = dict_.ntokens;
            var localTokenCount = 0L;
            var line = new List<int>();
            var labels = new List<int>();

            while (tokenCount_ < args_.epoch * ntokens)
            {
                var progress = (float)tokenCount_ / (args_.epoch * ntokens);
                var lr = (float)args_.lr * (1f - progress);

                if (args_.model == ModelName.sup)
                {
                    localTokenCount += dict_.GetLine(ifs, line, labels);
                    Supervised(state, lr, line.ToArray(), labels.ToArray());
                }
                else if (args_.model == ModelName.cbow)
                {
                    localTokenCount += dict_.GetLine(ifs, line, state.rng);
                    Cbow(state, lr, line.ToArray());
                }
                else if (args_.model == ModelName.sg)
                {
                    localTokenCount += dict_.GetLine(ifs, line, state.rng);
                    Skipgram(state, lr, line.ToArray());
                }

                if (localTokenCount > args_.lrUpdateRate)
                {
                    tokenCount_ += localTokenCount;
                    localTokenCount = 0;

                    if (threadId == 0 && args_.verbose > 1)
                    {
                        loss_ = state.GetLoss();
                    }
                }
            }

            if (threadId == 0)
            {
                loss_ = state.GetLoss();
            }

            ifs.Close();
        }

        protected List<Tuple<float, string>> GetNN(
            DenseMatrix wordVectors,
            Vector query,
            int k,
            OrderedSet<string> banSet)
        {
            var heap = new OrderedBag<Tuple<float, string>>();

            var queryNorm = query.Norm();
            if (Math.Abs(queryNorm) < 1e-8)
            {
                queryNorm = 1;
            }

            for (int i = 0; i < dict_.nwords; i++)
            {
                var word = dict_.GetWord(i);
                if (banSet.GetLast() == word)
                {
                    var dp = wordVectors.DotRow(query.Data, i);
                    var similarity = dp / queryNorm;

                    if (heap.Count == k && similarity < heap.GetFirst().Item1)
                    {
                        continue;
                    }

                    heap.Add(Tuple.Create(similarity, word));

                    if (heap.Count > k)
                    {
                        heap.RemoveFirst();
                    }
                }
            }

            return heap.ToList();
        }

        protected void LazyComputeWordVectors()
        {
            if (wordVectors_ == null)
            {
                wordVectors_ = new DenseMatrix(
                    new DenseMatrix(dict_.nwords, args_.dim));
                PrecomputeWordVectors(wordVectors_);
            }
        }

        protected void PrintInfo(float progress, float loss, TextWriter log_stream)
        {
            var end = DateTime.Now.TimeOfDay;
            var t = (float)(end - start_).TotalMilliseconds;
            var lr = args_.lr * (1f - progress);
            var wst = 0L;

            var eta = 2592000L;

            if (progress > 0 && t >= 0)
            {
                progress = progress * 100;
                eta = (long)(t * (100 - progress) / progress);
                wst = (long)(tokenCount_ / t / args_.thread);
            }

            var etah = eta / 3600;
            var etam = (eta % 3600) / 60;

            log_stream.Write(
                $"Progress: {progress,5:0.#}% words/sec/thread: {wst,7} " +
                $"lr: {lr,9:0.######} loss: {loss:9,0.######} " +
                $"ETA: {etah,3}h{etam,2}m");
        }

        protected Matrix GetInputMatrixFromFile(string filename)
        {
            var file = new FileStream(filename, FileMode.Open, FileAccess.Read);
            var reader = new BinaryReader(file);
            var words = new List<string>();
            long n, dim;
            
            if (!file.CanRead) {
                throw new ArgumentException($"{filename} cannot be opened for loading!");
            }

            n = reader.ReadInt64();
            dim = reader.ReadInt64();

            if (dim != args_.dim)
            {
                throw new ArgumentException($"Dimension of pretrained vectors ({dim}) does not match dimension ({args_.dim})!");
            }

            var mat = new DenseMatrix(n, dim);
            for (int i = 0; i < n; i++)
            {
                var word = reader.ReadString();

                words.Add(word);
                dict_.Add(word);
                for (int j = 0; j < dim; j++)
                {
                    mat[i, j] = reader.ReadSingle();
                }
            }
            reader.Close();

            dict_.Threshold(1, 0);
            dict_.Init();
            var input = new DenseMatrix(dict_.nwords + args_.bucket, args_.dim);
            input.Uniform(1f / args_.dim);

            for (int i = 0; i < n; i++)
            {
                var idx = dict_.GetId(words[i]);
                if (idx < 0 || idx >= dict_.nwords)
                {
                    continue;
                }

                for (int j = 0; j < dim; j++)
                {
                    input[idx, j] = mat[i, j];
                }
            }
            return input;
        }

        protected Matrix CreateRandomMatrix()
        {
            var input = new DenseMatrix(dict_.nwords + args_.bucket, args_.dim);
            input.Uniform(1f / args_.dim);

            return input;
        }

        protected Matrix CreateTrainOutputMatrix()
        {
            var m = (args_.model == ModelName.sup) ? dict_.nlabels : dict_.nwords;
            var output = new DenseMatrix(m, args_.dim);
            output.Zero();

            return output;
        }

        protected long[] GetTargetCounts()
        {
            if (args_.model == ModelName.sup)
            {
                return dict_.GetCounts(Dictionary.EntryType.label).ToArray();
            }
            else
            {
                return dict_.GetCounts(Dictionary.EntryType.word).ToArray();
            }
        }

        protected Loss CreateLoss(Matrix output)
        {
            var lossName = args_.loss;
            switch (lossName)
            {
                case LossName.hs:
                    return new HierarchicalSoftmaxLoss(output, GetTargetCounts());
                case LossName.ns:
                    return new NegativeSamplingLoss(output, args_.neg, GetTargetCounts());
                case LossName.softmax:
                    return new SoftmaxLoss(output);
                case LossName.ova:
                    return new OneVsAllLoss(output);
                default:
                    throw new InvalidOperationException("Unknown loss");
            }
        }

        protected void Supervised(
            Model.State state,
            float lr,
            int[] line,
            int[] labels)
        {
            if (labels.Length == 0 || line.Length == 0)
            {
                return;
            }

            if (args_.loss == LossName.ova)
            {
                model_.Update(line, labels, Model.kAllLabelsAsTarget, lr, state);
            }

            else
            {
                var i = state.rng.Next(0, labels.Length - 1);
                model_.Update(line, labels, i, lr, state);
            }
        }

        protected void Cbow(Model.State state, float lr, int[] line)
        {
            var bow = new List<int>();
            for (int w = 0; w < line.Length; w++)
            {
                var boundary = state.rng.Next(1, args_.ws);
                bow.Clear();

                for (int c = -boundary; c <= boundary; c++)
                {
                    if (c != 0 && w + c >= 0 && w + c < line.Length)
                    {
                        var ngrams = dict_.GetSubwords(line[w + c]);
                        bow.AddRange(ngrams);
                    }
                }
                model_.Update(bow.ToArray(), line, w, lr, state);
            }
        }

        protected void Skipgram(Model.State state, float lr, int[] line)
        {
            for (int w = 0; w < line.Length; w++)
            {
                var boundary = state.rng.Next(1, args_.ws);
                var ngrams = dict_.GetSubwords(line[w]);
                for (int c = -boundary; c <= boundary; c++)
                {
                    if (c != 0 && w + c >= 0 && w + c < line.Length)
                    {
                        model_.Update(ngrams, line, w + c, lr, state);
                    }
                }
            }
        }

        public FastText()
        {
            quant_ = false;
            wordVectors_ = null;
        }

        public int GetWordId(string word)
        {
            return dict_.GetId(word);
        }

        public int GetSubwordId(string subword)
        {
            var h = (int)(dict_.Hash(subword) % args_.bucket);
            return dict_.nwords + h;
        }

        public void GetWordVector(Vector vec, string word)
        {
            var ngrams = dict_.GetSubwords(word);
            vec.Zero();

            for (int i = 0; i < ngrams.Length; i++)
            {
                AddInputVector(vec, ngrams[i]);
            }

            if (ngrams.Length > 0)
            {
                vec.mul(1f / ngrams.Length);
            }
        }

        public void GetSubwordVector(Vector vec, string subword)
        {
            vec.Zero();
            var h = dict_.Hash(subword) % args_.bucket;
            h = h + dict_.nwords;
            AddInputVector(vec, (int)h);
        }

        public void GetInputVector(Vector vec, int ind)
        {
            vec.Zero();
            AddInputVector(vec, ind);
        }

        public Args GetArgs()
        {
            return args_;
        }

        public Dictionary GetDictionary()
        {
            return dict_;
        }

        public DenseMatrix GetInputMatrix()
        {
            if (quant_)
            {
                throw new InvalidOperationException("Can't export quantized matrix");
            }
            Debug.Assert(input_ != null);
            return input_ as DenseMatrix;
        }

        public DenseMatrix GetOutputMatrix()
        {
            if (quant_ && args_.qout)
            {
                throw new InvalidOperationException("Can't export quantized matrix");
            }
            Debug.Assert(output_ != null);
            return output_ as DenseMatrix;
        }

        public void SaveVectors(string filename)
        {
            var ofs = new FileStream(filename, FileMode.OpenOrCreate, FileAccess.Write);
            var writer = new StreamWriter(ofs);

            if (!ofs.CanWrite)
            {
                throw new ArgumentException($"{filename} cannot be opened for saving vectors!");
            }

            writer.WriteLine($"{dict_.nwords} {args_.dim}");

            var vec = new Vector(args_.dim);
            for (int i = 0; i < dict_.nwords; i++)
            {
                var word = dict_.GetWord(i);
                GetWordVector(vec, word);
                writer.WriteLine($"{word} {vec}");
            }
            writer.Close();
        }

        public void SaveModel(string filename)
        {
            var ofs = new FileStream(filename, FileMode.OpenOrCreate, FileAccess.Write);
            var writer = new BinaryWriter(ofs);

            if (!ofs.CanWrite)
            {
                throw new ArgumentException($"{filename} cannot be opened for saving!");
            }

            SignModel(writer);
            args_.Save(writer);
            dict_.Save(writer);

            writer.Write(quant_);
            input_.Save(writer);

            writer.Write(args_.qout);
            output_.Save(writer);

            writer.Close();
        }

        public void SaveOutput(string filename)
        {
            var ofs = new FileStream(filename, FileMode.OpenOrCreate, FileAccess.Write);
            var writer = new StreamWriter(ofs);

            if (!ofs.CanWrite)
            {
                throw new ArgumentException($"{filename} cannot be opened for saving vectors!");
            }

            if (quant_)
            {
                throw new ArgumentException("Option -saveOutput is not supported for quantized models.");
            }

            var n = (args_.model == ModelName.sup) ? dict_.nlabels : dict_.nwords;

            writer.WriteLine($"{n} {args_.dim}");
            var vec = new Vector(args_.dim);
            for (int i = 0; i < n; i++)
            {
                var word = (args_.model == ModelName.sup) ? dict_.GetLabel(i) : dict_.GetWord(i);

                vec.Zero();
                vec.AddRow(output_, i);
                writer.WriteLine($"{word} {vec}");
            }
            writer.Close();
        }

        public void LoadModel(BinaryReader reader)
        {
            args_ = new Args();
            input_ = new DenseMatrix();
            output_ = new DenseMatrix();
            args_.Load(reader);

            if (version == 11 && args_.model == ModelName.sup)
            {
                // backward compatibility: old supervised models do not use char ngrams.
                args_.maxn = 0;
            }
            dict_ = new Dictionary(args_, reader);

            var  quant_input = reader.ReadBoolean();
            if (quant_input)
            {
                quant_ = true;
                input_ = new QuantMatrix();
            }
            input_.Load(reader);

            if (!quant_input && dict_.IsPruned())
            {
                throw new ArgumentException(
                    "Invalid model file.\n" +
                    "Please download the updated model from www.fasttext.cc.\n" +
                    "See issue #332 on Github for more information.\n");
            }

            args_.qout = reader.ReadBoolean();
            if (quant_ && args_.qout)
            {
                output_ = new QuantMatrix();
            }
            output_.Load(reader);

            var loss = CreateLoss(output_);
            var normalizeGradient = (args_.model == ModelName.sup);
            model_ = new Model(input_, output_, loss, normalizeGradient);
        }

        public void LoadModel(string filename)
        {
            var ifs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            var reader = new BinaryReader(ifs);

            if (!ifs.CanRead)
            {
                throw new ArgumentException($"{filename} cannot be opened for loading!");
            }

            if (!CheckModel(reader))
            {
                throw new ArgumentException($"{filename} has wrong file format!");
            }

            LoadModel(reader);
            reader.Close();
        }

        public void GetSentenceVector(Stream stream, Vector svec)
        {
            svec.Zero();
            if (args_.model == ModelName.sup)
            {
                var line = new List<int>();
                var labels = new List<int>();
                dict_.GetLine(stream, line, labels);

                for (int i = 0; i < line.Count; i++)
                {
                    AddInputVector(svec, line[i]);
                }

                if (line.Count != 0)
                {
                    svec.mul(1f / line.Count);
                }
            }
            else
            {
                var vec = new Vector(args_.dim);
                var reader = new StreamReader(stream);
                var sentence = reader.ReadLine();

                int count = 0;
                while (!reader.EndOfStream)
                {
                    var words = reader.ReadLine().Split(' ');

                    foreach (var word in words)
                    {
                        GetWordVector(vec, word);
                        var norm = vec.Norm();
                        if (norm > 0)
                        {
                            vec.mul(1f / norm);
                            svec.AddVector(vec);
                            count++;
                        }
                    }
                }

                if (count > 0)
                {
                    svec.mul(1f / count);
                }
            }
        }

        public void Quantize(Args qargs)
        {
            if (args_.model != ModelName.sup)
            {
                throw new ArgumentException("For now we only support quantization of supervised models");
            }

            args_.input = qargs.input;
            args_.qout = qargs.qout;
            args_.output = qargs.output;
            var input = input_ as DenseMatrix;
            var output = output_ as DenseMatrix;
            bool normalizeGradient = (args_.model == ModelName.sup);

            if (qargs.cutoff > 0 && qargs.cutoff < input.Size(0))
            {
                var idx = SelectEmbeddings(qargs.cutoff);
                dict_.Prune(idx);
                var ninput = new DenseMatrix(idx.Count, args_.dim);
                for (int i = 0; i < idx.Count; i++)
                {
                    for (int j = 0; j < args_.dim; j++)
                    {
                        ninput[i, j] = input[idx[i], j];
                    }
                }
                input = ninput;
                if (qargs.retrain)
                {
                    args_.epoch = qargs.epoch;
                    args_.lr = qargs.lr;
                    args_.thread = qargs.thread;
                    args_.verbose = qargs.verbose;
                    var loss1 = CreateLoss(output_);
                    model_ = new Model(input, output, loss1, normalizeGradient);
                    StartThreads();
                }
            }

            input_ = new QuantMatrix(input, qargs.dsub, qargs.qnorm);

            if (args_.qout)
            {
                output_ = new QuantMatrix(input, 2, qargs.qnorm);
            }

            quant_ = true;
            var loss = CreateLoss(output_);
            model_ = new Model(input_, output_, loss, normalizeGradient);
        }

        public Tuple<long, double, double> Test(Stream stream, int k, float threshold = 0f)
        {
            var meter = new Meter();
            Test(stream, k, threshold, meter);

            return Tuple.Create(meter.nexamples, meter.Precision(), meter.Recall());
        }

        public void Test(Stream stream, int k, float threshold, Meter meter)
        {
            var line = new List<int>();
            var labels = new List<int>();
            var predictions = new Predictions();

            while (stream.Position != stream.Length)
            {
                line.Clear();
                labels.Clear();
                dict_.GetLine(stream, line, labels);

                if (labels.Count != 0 && line.Count != 0)
                {
                    predictions.Clear();
                    Predict(k, line.ToArray(), predictions, threshold);
                    meter.Log(labels.ToArray(), predictions);
                }
            }
        }

        public void Predict(int k, int[] words, Predictions predictions, float threshold = 0f)
        {
            if (words.Length == 0)
            {
                return;
            }

            var state = new Model.State(args_.dim, dict_.nlabels, 0);

            if (args_.model != ModelName.sup)
            {
                throw new ArgumentException("Model needs to be supervised for prediction!");
            }

            model_.Predict(words, k, threshold, predictions, state);
        }

        public bool PredictLine(
            Stream stream,
            List<Tuple<float, string>> predictions,
            int k,
            float threshold)
        {
            predictions.Clear();

            if (stream.Position == stream.Length)
            {
                return false;
            }

            var words = new List<int>();
            var labels = new List<int>();

            dict_.GetLine(stream, words, labels);

            var linePredictions = new Predictions();
            Predict(k, words.ToArray(), linePredictions, threshold);

            foreach (var p in linePredictions)
            {
                predictions.Add(Tuple.Create((float)Math.Exp(p.Item1), dict_.GetLabel(p.Item2)));
            }

            return true;
        }

        public List<Tuple<string, Vector>> GetNgramVectors(string word)
        {
            var result = new List<Tuple<string, Vector>>();
            var ngrams = new List<int>();
            var substrings = new List<string>();
            dict_.GetSubwords(word, ngrams, substrings);

            Debug.Assert(ngrams.Count <= substrings.Count);

            for (int i = 0; i < ngrams.Count; i++)
            {
                var vec = new Vector(args_.dim);

                if (ngrams[i] >= 0)
                {
                    vec.AddRow(input_, ngrams[i]);
                }

                result.Add(Tuple.Create(substrings[i], vec));
            }

            return result;
        }

        public List<Tuple<float, string>> GetNN(string word, int k)
        {
            var query = new Vector(args_.dim);

            GetWordVector(query, word);

            LazyComputeWordVectors();

            Debug.Assert(wordVectors_ != null);

            return GetNN(wordVectors_, query, k, new OrderedSet<string> { word });
        }

        public List<Tuple<float, string>> GetAnalogies(int k, string wordA, string wordB, string wordC)
        {
            var query = new Vector(args_.dim);
            query.Zero();

            var buffer = new Vector(args_.dim);
            GetWordVector(buffer, wordA);
            query.AddVector(buffer, 1f / (buffer.Norm() + 1e-8f));
            GetWordVector(buffer, wordB);
            query.AddVector(buffer, -1f / (buffer.Norm() + 1e-8f));
            GetWordVector(buffer, wordC);
            query.AddVector(buffer, 1f / (buffer.Norm() + 1e-8f));

            LazyComputeWordVectors();

            Debug.Assert(wordVectors_ != null);

            return GetNN(wordVectors_, query, k, new OrderedSet<string> { wordA, wordB, wordC });
        }

        public void Train(Args args)
        {
            args_ = args;
            dict_ = new Dictionary(args_);

            if (args_.input == "-")
            {
                // manage expectations
                throw new ArgumentException("Cannot use stdin for training!");
            }

            var ifs = new FileStream(args_.input, FileMode.Open, FileAccess.Read);
            if (!ifs.CanRead)
            {
                throw new ArgumentException($"{args_.input} cannot be opened for training!");
            }
            dict_.ReadFromFile(ifs);
            ifs.Close();

            if (!string.IsNullOrEmpty(args_.pretrainedVectors))
            {
                input_ = GetInputMatrixFromFile(args_.pretrainedVectors);
            }
            else
            {
                input_ = CreateRandomMatrix();
            }

            output_ = CreateTrainOutputMatrix();
            var loss = CreateLoss(output_);
            bool normalizeGradient = (args_.model == ModelName.sup);
            model_ = new Model(input_, output_, loss, normalizeGradient);
            StartThreads();
        }

        public int GetDimension()
        {
            return args_.dim;
        }

        public bool IsQuant()
        {
            return quant_;
        }

        [Obsolete("SelectEmbeddings is being deprecated.")]
        public List<int> SelectEmbeddings(int cutoff)
        {
            var input = input_ as DenseMatrix;
            var norms = new Vector(input.Size(0));
            input.L2NormRow(norms.Data);
            var idx = new List<int>();

            for (int i = 0; i < input.Size(0); i++)
            {
                idx.Add(i);
            }

            var eosid = dict_.GetId(Dictionary.EOS);
            idx.Sort(new Comparison<int>((i1, i2) =>
            {
                var b = eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
                return b ? 1 : 0;
            }));
            idx = idx.Take(cutoff).ToList();
            return idx;
        }

        [Obsolete("PrecomputeWordVectors is being deprecated.")]
        public void PrecomputeWordVectors(DenseMatrix wordVectors)
        {
            var vec = new Vector(args_.dim);
            wordVectors.Zero();

            for (int i = 0; i < dict_.nwords; i++)
            {
                var word = dict_.GetWord(i);
                GetWordVector(vec, word);

                var norm = vec.Norm();
                if (norm > 0)
                {
                    wordVectors.AddVectorToRow(vec.Data, i, 1f / norm);
                }
            }
        }
    }
}
