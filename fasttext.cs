using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Wintellect.PowerCollections;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

#pragma warning disable CS0618 // Type or member is obsolete

namespace fasttext
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

        protected void signModel(BinaryWriter writer)
        {
            writer.Write(FASTTEXT_FILEFORMAT_MAGIC_INT32);
            writer.Write(FASTTEXT_VERSION);
        }

        protected bool checkModel(BinaryReader reader)
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

        protected void startThreads()
        {
            start_ = DateTime.Now.TimeOfDay;
            tokenCount_ = 0;
            loss_ = -1;

            var threads = new List<Thread>();
            for (int i = 0; i < args_.thread; i++)
            {
                threads.Add(new Thread(() => trainThread(i)));
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
                    printInfo(progress, loss_, Console.Error);
                }
            }

            for (int i = 0; i < args_.thread; i++)
            {
                threads[i].Join();
            }

            if (args_.verbose > 0)
            {
                Console.Error.Write("\r");
                printInfo(1f, loss_, Console.Error);
                Console.Error.Write(Environment.NewLine);
            }
        }

        protected void addInputVector(Vector vec, int ind)
        {
            vec.addRow(input_, ind);
        }

        protected void trainThread(int threadId)
        {
            var ifs = new FileStream(args_.input, FileMode.Open, FileAccess.Read);
            ifs.Flush();
            ifs.Seek(threadId * ifs.Length / args_.thread, SeekOrigin.Begin);

            var state = new Model.State(args_.dim, (int)output_.size(0), threadId);

            var ntokens = dict_.ntokens;
            var localTokenCount = 0L;
            var line = new List<int>();
            var labels = new List<int>();

            while (tokenCount_ < args_.epoch * ntokens)
            {
                var progress = (float)tokenCount_ / (args_.epoch * ntokens);
                var lr = (float)args_.lr * (1f - progress);

                if (args_.model == model_name.sup)
                {
                    localTokenCount += dict_.getLine(ifs, line, labels);
                    supervised(state, lr, line.ToArray(), labels.ToArray());
                }
                else if (args_.model == model_name.cbow)
                {
                    localTokenCount += dict_.getLine(ifs, line, state.rng);
                    cbow(state, lr, line.ToArray());
                }
                else if (args_.model == model_name.sg)
                {
                    localTokenCount += dict_.getLine(ifs, line, state.rng);
                    skipgram(state, lr, line.ToArray());
                }

                if (localTokenCount > args_.lrUpdateRate)
                {
                    tokenCount_ += localTokenCount;
                    localTokenCount = 0;

                    if (threadId == 0 && args_.verbose > 1)
                    {
                        loss_ = state.getLoss();
                    }
                }
            }

            if (threadId == 0)
            {
                loss_ = state.getLoss();
            }

            ifs.Close();
        }

        protected List<Tuple<float, string>> getNN(
            DenseMatrix wordVectors,
            Vector query,
            int k,
            OrderedSet<string> banSet)
        {
            var heap = new OrderedBag<Tuple<float, string>>();

            var queryNorm = query.norm();
            if (Math.Abs(queryNorm) < 1e-8)
            {
                queryNorm = 1;
            }

            for (int i = 0; i < dict_.nwords; i++)
            {
                var word = dict_.getWord(i);
                if (banSet.GetLast() == word)
                {
                    var dp = wordVectors.dotRow(query.data, i);
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

        protected void lazyComputeWordVectors()
        {
            if (wordVectors_ == null)
            {
                wordVectors_ = new DenseMatrix(
                    new DenseMatrix(dict_.nwords, args_.dim));
                precomputeWordVectors(wordVectors_);
            }
        }

        protected void printInfo(float progress, float loss, TextWriter log_stream)
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

        protected Matrix getInputMatrixFromFile(string filename)
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
                dict_.add(word);
                for (int j = 0; j < dim; j++)
                {
                    mat[i, j] = reader.ReadSingle();
                }
            }
            reader.Close();

            dict_.threshold(1, 0);
            dict_.init();
            var input = new DenseMatrix(dict_.nwords + args_.bucket, args_.dim);
            input.uniform(1f / args_.dim);

            for (int i = 0; i < n; i++)
            {
                var idx = dict_.getId(words[i]);
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

        protected Matrix createRandomMatrix()
        {
            var input = new DenseMatrix(dict_.nwords + args_.bucket, args_.dim);
            input.uniform(1f / args_.dim);

            return input;
        }

        protected Matrix createTrainOutputMatrix()
        {
            var m = (args_.model == model_name.sup) ? dict_.nlabels : dict_.nwords;
            var output = new DenseMatrix(m, args_.dim);
            output.zero();

            return output;
        }

        protected long[] getTargetCounts()
        {
            if (args_.model == model_name.sup)
            {
                return dict_.getCounts(Dictionary.entry_type.label).ToArray();
            }
            else
            {
                return dict_.getCounts(Dictionary.entry_type.word).ToArray();
            }
        }

        protected Loss createLoss(Matrix output)
        {
            var lossName = args_.loss;
            switch (lossName)
            {
                case loss_name.hs:
                    return new HierarchicalSoftmaxLoss(output, getTargetCounts());
                case loss_name.ns:
                    return new NegativeSamplingLoss(output, args_.neg, getTargetCounts());
                case loss_name.softmax:
                    return new SoftmaxLoss(output);
                case loss_name.ova:
                    return new OneVsAllLoss(output);
                default:
                    throw new InvalidOperationException("Unknown loss");
            }
        }

        protected void supervised(
            Model.State state,
            float lr,
            int[] line,
            int[] labels)
        {
            if (labels.Length == 0 || line.Length == 0)
            {
                return;
            }

            if (args_.loss == loss_name.ova)
            {
                model_.update(line, labels, Model.kAllLabelsAsTarget, lr, state);
            }

            else
            {
                var i = state.rng.Next(0, labels.Length - 1);
                model_.update(line, labels, i, lr, state);
            }
        }

        protected void cbow(Model.State state, float lr, int[] line)
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
                        var ngrams = dict_.getSubwords(line[w + c]);
                        bow.AddRange(ngrams);
                    }
                }
                model_.update(bow.ToArray(), line, w, lr, state);
            }
        }

        protected void skipgram(Model.State state, float lr, int[] line)
        {
            for (int w = 0; w < line.Length; w++)
            {
                var boundary = state.rng.Next(1, args_.ws);
                var ngrams = dict_.getSubwords(line[w]);
                for (int c = -boundary; c <= boundary; c++)
                {
                    if (c != 0 && w + c >= 0 && w + c < line.Length)
                    {
                        model_.update(ngrams, line, w + c, lr, state);
                    }
                }
            }
        }

        public FastText()
        {
            quant_ = false;
            wordVectors_ = null;
        }

        public int getWordId(string word)
        {
            return dict_.getId(word);
        }

        public int getSubwordId(string subword)
        {
            var h = (int)(dict_.hash(subword) % args_.bucket);
            return dict_.nwords + h;
        }

        public void getWordVector(Vector vec, string word)
        {
            var ngrams = dict_.getSubwords(word);
            vec.zero();

            for (int i = 0; i < ngrams.Length; i++)
            {
                addInputVector(vec, ngrams[i]);
            }

            if (ngrams.Length > 0)
            {
                vec.mul(1f / ngrams.Length);
            }
        }

        public void getSubwordVector(Vector vec, string subword)
        {
            vec.zero();
            var h = dict_.hash(subword) % args_.bucket;
            h = h + dict_.nwords;
            addInputVector(vec, (int)h);
        }

        public void getInputVector(Vector vec, int ind)
        {
            vec.zero();
            addInputVector(vec, ind);
        }

        public Args getArgs()
        {
            return args_;
        }

        public Dictionary getDictionary()
        {
            return dict_;
        }

        public DenseMatrix getInputMatrix()
        {
            if (quant_)
            {
                throw new InvalidOperationException("Can't export quantized matrix");
            }
            Debug.Assert(input_ != null);
            return input_ as DenseMatrix;
        }

        public DenseMatrix getOutputMatrix()
        {
            if (quant_ && args_.qout)
            {
                throw new InvalidOperationException("Can't export quantized matrix");
            }
            Debug.Assert(output_ != null);
            return output_ as DenseMatrix;
        }

        public void saveVectors(string filename)
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
                var word = dict_.getWord(i);
                getWordVector(vec, word);
                writer.WriteLine($"{word} {vec}");
            }
            writer.Close();
        }

        public void saveModel(string filename)
        {
            var ofs = new FileStream(filename, FileMode.OpenOrCreate, FileAccess.Write);
            var writer = new BinaryWriter(ofs);

            if (!ofs.CanWrite)
            {
                throw new ArgumentException($"{filename} cannot be opened for saving!");
            }

            signModel(writer);
            args_.save(writer);
            dict_.save(writer);

            writer.Write(quant_);
            input_.save(writer);

            writer.Write(args_.qout);
            output_.save(writer);

            writer.Close();
        }

        public void saveOutput(string filename)
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

            var n = (args_.model == model_name.sup) ? dict_.nlabels : dict_.nwords;

            writer.WriteLine($"{n} {args_.dim}");
            var vec = new Vector(args_.dim);
            for (int i = 0; i < n; i++)
            {
                var word = (args_.model == model_name.sup) ? dict_.getLabel(i) : dict_.getWord(i);

                vec.zero();
                vec.addRow(output_, i);
                writer.WriteLine($"{word} {vec}");
            }
            writer.Close();
        }

        public void loadModel(BinaryReader reader)
        {
            args_ = new Args();
            input_ = new DenseMatrix();
            output_ = new DenseMatrix();
            args_.load(reader);

            if (version == 11 && args_.model == model_name.sup)
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
            input_.load(reader);

            if (!quant_input && dict_.isPruned())
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
            output_.load(reader);

            var loss = createLoss(output_);
            var normalizeGradient = (args_.model == model_name.sup);
            model_ = new Model(input_, output_, loss, normalizeGradient);
        }

        public void loadModel(string filename)
        {
            var ifs = new FileStream(filename, FileMode.Open, FileAccess.Read);
            var reader = new BinaryReader(ifs);

            if (!ifs.CanRead)
            {
                throw new ArgumentException($"{filename} cannot be opened for loading!");
            }

            if (!checkModel(reader))
            {
                throw new ArgumentException($"{filename} has wrong file format!");
            }

            loadModel(reader);
            reader.Close();
        }

        public void getSentenceVector(Stream stream, Vector svec)
        {
            svec.zero();
            if (args_.model == model_name.sup)
            {
                var line = new List<int>();
                var labels = new List<int>();
                dict_.getLine(stream, line, labels);

                for (int i = 0; i < line.Count; i++)
                {
                    addInputVector(svec, line[i]);
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
                        getWordVector(vec, word);
                        var norm = vec.norm();
                        if (norm > 0)
                        {
                            vec.mul(1f / norm);
                            svec.addVector(vec);
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

        public void quantize(Args qargs)
        {
            if (args_.model != model_name.sup)
            {
                throw new ArgumentException("For now we only support quantization of supervised models");
            }

            args_.input = qargs.input;
            args_.qout = qargs.qout;
            args_.output = qargs.output;
            var input = input_ as DenseMatrix;
            var output = output_ as DenseMatrix;
            bool normalizeGradient = (args_.model == model_name.sup);

            if (qargs.cutoff > 0 && qargs.cutoff < input.size(0))
            {
                var idx = selectEmbeddings(qargs.cutoff);
                dict_.prune(idx);
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
                    var loss1 = createLoss(output_);
                    model_ = new Model(input, output, loss1, normalizeGradient);
                    startThreads();
                }
            }

            input_ = new QuantMatrix(input, qargs.dsub, qargs.qnorm);

            if (args_.qout)
            {
                output_ = new QuantMatrix(input, 2, qargs.qnorm);
            }

            quant_ = true;
            var loss = createLoss(output_);
            model_ = new Model(input_, output_, loss, normalizeGradient);
        }

        public Tuple<long, double, double> test(Stream stream, int k, float threshold = 0f)
        {
            var meter = new Meter();
            test(stream, k, threshold, meter);

            return Tuple.Create(meter.nexamples, meter.precision(), meter.recall());
        }

        public void test(Stream stream, int k, float threshold, Meter meter)
        {
            var line = new List<int>();
            var labels = new List<int>();
            var predictions = new Predictions();

            while (stream.Position != stream.Length)
            {
                line.Clear();
                labels.Clear();
                dict_.getLine(stream, line, labels);

                if (labels.Count != 0 && line.Count != 0)
                {
                    predictions.Clear();
                    predict(k, line.ToArray(), predictions, threshold);
                    meter.log(labels.ToArray(), predictions);
                }
            }
        }

        public void predict(int k, int[] words, Predictions predictions, float threshold = 0f)
        {
            if (words.Length == 0)
            {
                return;
            }

            var state = new Model.State(args_.dim, dict_.nlabels, 0);

            if (args_.model != model_name.sup)
            {
                throw new ArgumentException("Model needs to be supervised for prediction!");
            }

            model_.predict(words, k, threshold, predictions, state);
        }

        public bool predictLine(
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

            dict_.getLine(stream, words, labels);

            var linePredictions = new Predictions();
            predict(k, words.ToArray(), linePredictions, threshold);

            foreach (var p in linePredictions)
            {
                predictions.Add(Tuple.Create((float)Math.Exp(p.Item1), dict_.getLabel(p.Item2)));
            }

            return true;
        }

        public List<Tuple<string, Vector>> getNgramVectors(string word)
        {
            var result = new List<Tuple<string, Vector>>();
            var ngrams = new List<int>();
            var substrings = new List<string>();
            dict_.getSubwords(word, ngrams, substrings);

            Debug.Assert(ngrams.Count <= substrings.Count);

            for (int i = 0; i < ngrams.Count; i++)
            {
                var vec = new Vector(args_.dim);

                if (ngrams[i] >= 0)
                {
                    vec.addRow(input_, ngrams[i]);
                }

                result.Add(Tuple.Create(substrings[i], vec));
            }

            return result;
        }

        public List<Tuple<float, string>> getNN(string word, int k)
        {
            var query = new Vector(args_.dim);

            getWordVector(query, word);

            lazyComputeWordVectors();

            Debug.Assert(wordVectors_ != null);

            return getNN(wordVectors_, query, k, new OrderedSet<string> { word });
        }

        public List<Tuple<float, string>> getAnalogies(int k, string wordA, string wordB, string wordC)
        {
            var query = new Vector(args_.dim);
            query.zero();

            var buffer = new Vector(args_.dim);
            getWordVector(buffer, wordA);
            query.addVector(buffer, 1f / (buffer.norm() + 1e-8f));
            getWordVector(buffer, wordB);
            query.addVector(buffer, -1f / (buffer.norm() + 1e-8f));
            getWordVector(buffer, wordC);
            query.addVector(buffer, 1f / (buffer.norm() + 1e-8f));

            lazyComputeWordVectors();

            Debug.Assert(wordVectors_ != null);

            return getNN(wordVectors_, query, k, new OrderedSet<string> { wordA, wordB, wordC });
        }

        public void train(Args args)
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
            dict_.readFromFile(ifs);
            ifs.Close();

            if (!string.IsNullOrEmpty(args_.pretrainedVectors))
            {
                input_ = getInputMatrixFromFile(args_.pretrainedVectors);
            }
            else
            {
                input_ = createRandomMatrix();
            }

            output_ = createTrainOutputMatrix();
            var loss = createLoss(output_);
            bool normalizeGradient = (args_.model == model_name.sup);
            model_ = new Model(input_, output_, loss, normalizeGradient);
            startThreads();
        }

        public int getDimension()
        {
            return args_.dim;
        }

        public bool isQuant()
        {
            return quant_;
        }

        [Obsolete("selectEmbeddings is being deprecated.")]
        public List<int> selectEmbeddings(int cutoff)
        {
            var input = input_ as DenseMatrix;
            var norms = new Vector(input.size(0));
            input.l2NormRow(norms.data);
            var idx = new List<int>();

            for (int i = 0; i < input.size(0); i++)
            {
                idx.Add(i);
            }

            var eosid = dict_.getId(Dictionary.EOS);
            idx.Sort(new Comparison<int>((i1, i2) =>
            {
                var b = eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
                return b ? 1 : 0;
            }));
            idx = idx.Take(cutoff).ToList();
            return idx;
        }

        [Obsolete("precomputeWordVectors is being deprecated.")]
        public void precomputeWordVectors(DenseMatrix wordVectors)
        {
            var vec = new Vector(args_.dim);
            wordVectors.zero();

            for (int i = 0; i < dict_.nwords; i++)
            {
                var word = dict_.getWord(i);
                getWordVector(vec, word);

                var norm = vec.norm();
                if (norm > 0)
                {
                    wordVectors.addVectorToRow(vec.data, i, 1f / norm);
                }
            }
        }
    }
}
