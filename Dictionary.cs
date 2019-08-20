using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace FastText
{
    public class Dictionary
    {
        public enum EntryType : byte { word = 0, label = 1 };

        public class Entry
        {
            public string word;
            public long count;
            public EntryType type;
            public List<int> subwords;
        };

        protected const int MAX_VOCAB_SIZE = 30000000;
        protected const int MAX_LINE_SIZE = 1024;

        protected Args args_;
        protected int[] word2int_;
        protected List<Entry> words_;

        protected float[] pdiscard_;
        protected int size_;
        protected int nwords_;
        protected int nlabels_;
        protected long ntokens_;

        protected long pruneidx_size_;
        protected Dictionary<int, int> pruneidx_;

        public const string EOS = "</s>";
        public const string BOW = "<";
        public const string EOW = ">";

        public int nwords => nwords_;
        public int nlabels => nlabels_;
        public long ntokens => ntokens_;

        public Dictionary(Args args)
        {
            args_ = args;

            word2int_ = new int[MAX_VOCAB_SIZE];
            Array.Fill(word2int_, -1);

            size_ = 0;
            nwords_ = 0;
            nlabels_ = 0;
            ntokens_ = 0;
            pruneidx_size_ = -1;

            words_ = new List<Entry>();
            pruneidx_ = new Dictionary<int, int>();
        }

        public Dictionary(Args args, BinaryReader reader)
            : this(args)
        {
            Load(reader);
        }

        protected int Find(string w)
        {
            return Find(w, Hash(w));
        }

        protected int Find(string w, uint h)
        {
            var word2intsize = word2int_.Length;
            var id = h % word2intsize;

            while (word2int_[id] != -1 && words_[word2int_[id]].word != w)
            {
                id = (id + 1) % word2intsize;
            }

            return (int)id;
        }

        public void Add(string w)
        {
            var h = Find(w);
            ntokens_++;
            if (word2int_[h] == -1)
            {
                var e = new Entry();
                e.word = w;
                e.count = 1;
                e.type = GetType(w);
                words_.Add(e);
                word2int_[h] = size_++;
            }
            else
            {
                var e = words_[word2int_[h]];
                e.count++;

                words_[word2int_[h]] = e;
            }
        }

        public int[] GetSubwords(int i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < nwords_);

            return words_[i].subwords.ToArray();
        }

        public int[] GetSubwords(string word)
        {
            var i = GetId(word);
            if (i >= 0)
            {
                return GetSubwords(i);
            }

            var ngrams = new List<int>();
            if (word != EOS)
            {
                ComputeSubwords(BOW + word + EOW, ngrams, null);
            }
            return ngrams.ToArray();
        }

        public void GetSubwords(string word, List<int> ngrams, List<string> substrings)
        {
            var i = GetId(word);
            ngrams.Clear();
            substrings.Clear();

            if (i >= 0)
            {
                ngrams.Add(i);
                substrings.Add(words_[i].word);
            }

            if (word != EOS)
            {
                ComputeSubwords(BOW + word + EOW, ngrams, substrings);
            }
        }

        public bool Discard(int id, float rand)
        {
            Debug.Assert(id >= 0);
            Debug.Assert(id < nwords_);

            if (args_.model == ModelName.sup)
            {
                return false;
            }

            return rand > pdiscard_[id];
        }

        public int GetId(string w, uint h)
        {
            var id = Find(w, h);
            return word2int_[id];
        }

        public int GetId(string w)
        {
            var id = Find(w);
            return word2int_[id];
        }

        public EntryType getType(int id)
        {
            Debug.Assert(id >= 0);
            Debug.Assert(id < size_);

            return words_[id].type;
        }

        public EntryType GetType(string w)
        {
            return (w.IndexOf(args_.label) == 0) ? EntryType.label : EntryType.word;
        }

        public string GetWord(int id)
        {
            Debug.Assert(id >= 0);
            Debug.Assert(id < size_);

            return words_[id].word;
        }

        public uint Hash(string str)
        {
            uint h = 2166136261;
            for (int i = 0; i < str.Length; i++)
            {
                h = h ^ ((uint)str[i]);
                h = h * 16777619;
            }
            return h;
        }

        public void ComputeSubwords(string word, List<int> ngrams, List<string> substrings)
        {
            for (int i = 0; i < word.Length; i++)
            {
                var ngram = string.Empty;

                if ((word[i] & 0xC0) == 0x80)
                {
                    continue;
                }

                for (int j = i, n = 1; j < word.Length && n <= args_.maxn; n++)
                {
                    ngram += word[j++];

                    while (j < word.Length && (word[j] & 0xC0) == 0x80)
                    {
                        ngram += word[j++];
                    }

                    if (n >= args_.minn && !(n == 1 && (i == 0 || j == word.Length)))
                    {
                        var h = Hash(ngram) % args_.bucket;
                        PushHash(ngrams, (int)h);

                        if (substrings != null)
                        {
                            substrings.Add(ngram);
                        }
                    }
                }
            }
        }

        protected void InitNgrams()
        {
            for (int i = 0; i < size_; i++)
            {
                var word = BOW + words_[i].word + EOW;
                words_[i].subwords = new List<int> { i };

                if (words_[i].word != EOS)
                {
                    ComputeSubwords(word, words_[i].subwords, null);
                }
            }
        }

        public bool ReadWord(Stream stream, out string word)
        {
            word = string.Empty;
            while (stream.Position != stream.Length)
            {
                var c = (char)stream.ReadByte();
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
                    c == '\f' || c == '\0')
                {
                    if (string.IsNullOrEmpty(word))
                    {
                        if (c == '\n')
                        {
                            word += EOS;
                            return true;
                        }
                        continue;
                    }
                    else
                    {
                        if (c == '\n')
                        {
                            stream.Position--;
                        }
                        return true;
                    }
                }
                word += c;
            }

            return !string.IsNullOrEmpty(word);
        }

        public void ReadFromFile(Stream stream)
        {
            var minThreshold = 1L;
            while (ReadWord(stream, out var word))
            {
                Add(word);

                if (ntokens_ % 1000000 == 0 && args_.verbose > 1)
                {
                    Console.Error.Write($"\rRead {(ntokens_ / 1000000)}M words");
                }

                if (size_ > 0.75 * MAX_VOCAB_SIZE)
                {
                    minThreshold++;
                    Threshold(minThreshold, minThreshold);
                }
            }
            Threshold(args_.minCount, args_.minCountLabel);
            InitTableDiscard();
            InitNgrams();
            if (args_.verbose > 0)
            {
                Console.Error.WriteLine($"\rRead {(ntokens_ / 1000000)}M words");
                Console.Error.WriteLine($"Number of words: {nwords_}");
                Console.Error.WriteLine($"Number of labels: {nlabels_}");
            }
            if (size_ == 0)
            {
                throw new ArgumentException("Empty vocabulary. Try a smaller -minCount value.");
            }
        }

        public void Threshold(long t, long tl)
        {
            words_.Sort(new Comparison<Entry>((e1, e2) =>
            {
                if (e1.type != e2.type)
                {
                    return e1.type < e2.type ? 1 : 0;
                }
                return e1.count > e2.count ? 1 : 0;
            }));

            words_ = words_.Where(e =>
            {
                return (e.type == EntryType.word && e.count >= t) ||
                (e.type == EntryType.label && e.count >= tl);
            }).ToList();

            size_ = 0;
            nwords_ = 0;
            nlabels_ = 0;

            Array.Fill(word2int_, -1);

            for (int i = 0; i < words_.Count; i++)
            {
                var e = words_[i];
                var h = Find(e.word);
                word2int_[h] = size_++;

                if (e.type == EntryType.word)
                {
                    nwords_++;
                }
                else if (e.type == EntryType.label)
                {
                    nlabels_++;
                }
            }
        }

        protected void InitTableDiscard()
        {
            pdiscard_ = new float[size_];

            for (int i = 0; i < size_; i++)
            {
                 var f = words_[i].count / (float)(ntokens_);
                pdiscard_[i] = (float)(Math.Sqrt(args_.t / f) + args_.t / f);
            }
        }

        public List<long> GetCounts(EntryType type)
        {
            var counts = new List<long>();
            for (int i = 0; i < size_; i++)
            {
                if (words_[i].type == type)
                {
                    counts.Add(words_[i].count);
                }
            }

            return counts;
        }

        protected void AddWordNgrams(List<int> line, int[] hashes, int n)
        {
            for (int i = 0; i < hashes.Length; i++)
            {
                long h = hashes[i];
                for (int j = i + 1; j < hashes.Length && j < i + n; j++)
                {
                    h = h * 116049371 + hashes[j];
                    PushHash(line, (int)(h % args_.bucket));
                }
            }
        }

        protected void AddSubwords(List<int> line, string token, int wid)
        {
            if (wid < 0)
            { 
                // out of vocab
                if (token != EOS)
                {
                    ComputeSubwords(BOW + token + EOW, line, null);
                }
            }
            else
            {
                if (args_.maxn <= 0)
                { 
                    // in vocab w/o subwords
                    line.Add(wid);
                }
                else
                { 
                    // in vocab w/ subwords
                    var ngrams = GetSubwords(wid);
                    line.AddRange(ngrams);
                }
            }
        }

        protected void Reset(Stream stream)
        {
            if (stream.Position == stream.Length)
            {
                stream.Flush();
                stream.Seek(0, SeekOrigin.Begin);
            }
        }

        public int GetLine(Stream stream, List<int> words, Random rng)
        {
            var ntokens = 0;

            Reset(stream);
            words.Clear();

            while (ReadWord(stream, out string token))
            {
                var h = Find(token);
                var wid = word2int_[h];

                if (wid < 0)
                {
                    continue;
                }

                ntokens++;

                if (getType(wid) == EntryType.word && !Discard(wid, (float)rng.NextDouble()))
                {
                    words.Add(wid);
                }

                if (ntokens > MAX_LINE_SIZE || token == EOS)
                {
                    break;
                }
            }
            return ntokens;
        }

        public int GetLine(Stream stream, List<int> words, List<int> labels)
        {
            var word_hashes = new List<int>();
            var ntokens = 0;

            Reset(stream);
            words.Clear();
            labels.Clear();

            while (ReadWord(stream, out var token))
            {
                var h = Hash(token);
                var wid = GetId(token, h);
                var type = wid < 0 ? GetType(token) : getType(wid);

                ntokens++;
                if (type == EntryType.word)
                {
                    AddSubwords(words, token, wid);
                    word_hashes.Add((int)h);
                }
                else if (type == EntryType.label && wid >= 0)
                {
                    labels.Add(wid - nwords_);
                }
                if (token == EOS)
                {
                    break;
                }
            }
            AddWordNgrams(words, word_hashes.ToArray(), args_.wordNgrams);
            return ntokens;
        }

        protected void PushHash(List<int> hashes, int id)
        {
            if (pruneidx_size_ == 0 || id < 0)
            {
                return;
            }

            if (pruneidx_size_ > 0)
            {
                if (pruneidx_.ContainsKey(id))
                {
                    id = pruneidx_[id];
                }
                else
                {
                    return;
                }
            }

            hashes.Add(nwords_ + id);
        }

        public string GetLabel(int lid)
        {
            if (lid < 0 || lid >= nlabels_)
            {
                throw new ArgumentException($"Label id is out of range [0, {nlabels_}]");
            }

            return words_[lid + nwords_].word;
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(size_);
            writer.Write(nwords_);
            writer.Write(nlabels_);
            writer.Write(ntokens_);
            writer.Write(pruneidx_size_);

            for (int i = 0; i < size_; i++)
            {
                Entry e = words_[i];
                var word_bytes = System.Text.Encoding.ASCII.GetBytes(e.word);
                writer.Write(word_bytes);
                writer.Write((byte)0);
                writer.Write(e.count);
                writer.Write((byte)e.type);
            }

            for (int i = 0; i < pruneidx_.Count; i++) {
                var kvp = pruneidx_.ElementAt(i);
                writer.Write(kvp.Key);
                writer.Write(kvp.Value);
            }
        }

        public void Load(BinaryReader reader)
        {
            words_.Clear();
            size_ = reader.ReadInt32();
            nwords_ = reader.ReadInt32();
            nlabels_ = reader.ReadInt32();
            ntokens_ = reader.ReadInt64();
            pruneidx_size_ = reader.ReadInt64();

            for (int i = 0; i < size_; i++)
            {
                var e = new Entry();
                var c = (char)reader.ReadByte();
                while (c != '\0')
                {
                    e.word += c;
                    c = (char)reader.ReadByte();
                }
                e.count = reader.ReadInt64();
                e.type = (EntryType)reader.ReadByte();
                words_.Add(e);
            }

            pruneidx_.Clear();

            for (int i = 0; i < pruneidx_size_; i++)
            {
                var key = reader.ReadInt32();
                var value = reader.ReadInt32();
                pruneidx_[key] = value;
            }

            InitTableDiscard();
            InitNgrams();

            var word2intsize = (int)Math.Ceiling(size_ / 0.7);
            word2int_ = Enumerable.Repeat(-1, word2intsize).ToArray();

            for (int i = 0; i < size_; i++)
            {
                word2int_[Find(words_[i].word)] = i;
            }
        }

        public void Dump(TextWriter writer)
        {
            writer.WriteLine($"{words_.Count}");

            for (int i = 0; i < words_.Count; i++)
            {
                var e = words_[i];
                var entryType = "word";

                if (e.type == EntryType.label)
                {
                    entryType = "label";
                }

                writer.WriteLine($"{e.word} {e.count} {entryType}");
            }
        }

        public void Init()
        {
            InitTableDiscard();
            InitNgrams();
        }

        public bool IsPruned()
        {
            return pruneidx_size_ >= 0;
        }

        public void Prune(List<int> idx)
        {
            var words = new List<int>();
            var ngrams = new List<int>();

            for (var i = 0; i < idx.Count; ++i)
            {
                var val = idx[i];
                if (val < nwords_)
                {
                    words.Add(val);
                }
                else
                {
                    ngrams.Add(val);
                }
            }

            words.Sort();
            idx = words;

            if (ngrams.Count != 0)
            {
                for (int i = 0; i < ngrams.Count; i++)
                {
                    var ngram = ngrams[i];
                    pruneidx_[ngram - nwords_] = i;
                }

                idx.AddRange(ngrams);
            }
            pruneidx_size_ = pruneidx_.Count;

            Array.Fill(word2int_, -1);

            int j = 0;
            for (int i = 0; i < words_.Count; i++)
            {
                if (getType(i) == EntryType.label ||
                    (j < words.Count && words[j] == i))
                {
                    words_[j] = words_[i];
                    word2int_[Find(words_[j].word)] = j;
                    j++;
                }
            }
            nwords_ = words.Count;
            size_ = nwords_ + nlabels_;
            words_.RemoveRange(size_, words_.Count - size_ - 1);
            InitNgrams();
        }
    }
}