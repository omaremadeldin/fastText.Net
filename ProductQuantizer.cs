using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FastText
{
    public class ProductQuantizer
    {
        protected const int nbits_ = 8;
        protected const int ksub_ = 1 << nbits_;
        protected const int max_points_per_cluster_ = 256;
        protected const int max_points_ = max_points_per_cluster_ * ksub_;
        protected const int seed_ = 1234;
        protected const int niter_ = 25;
        protected const float eps_ = 1E-7F;

        protected int dim_;
        protected int nsubq_;
        protected int dsub_;
        protected int lastdsub_;

        protected float[] centroids_;

        protected Random rng;

        private T[] SliceArray<T>(T[] arr, int index)
        {
            var arrLength = arr.Length - index;
            var result = new T[arrLength];
            Array.Copy(arr, index, result, 0, arrLength);
            return result;
        }

        private T[] ShuffleArray<T>(T[] arr)
        {
            int n = arr.Length;
            var result = new T[n];

            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = arr[k];
                result[k] = arr[n];
                result[n] = value;
            }

            return result;
        }

        public ProductQuantizer()
        {
        }

        public ProductQuantizer(int dim, int dsub)
        {
            dim_ = dim;
            nsubq_ = dim / dsub;
            dsub_ = dsub;
            centroids_ = new float[dim * ksub_];
            rng = new Random(seed_);
            lastdsub_ = dim % dsub;

            if (lastdsub_ == 0)
            {
                lastdsub_ = dsub_;
            }
            else
            {
                nsubq_++;
            }
        }

        public float[] GetCentroids(int m, byte i)
        {
            int index;

            if (m == nsubq_ - 1)
            {
                index = m * ksub_ * dsub_ + i * lastdsub_;
            }
            else
            {
                index = (m * ksub_ + i) * dsub_;
            }

            return SliceArray(centroids_, index);
        }

        public float AssignCentroid(float[] x, float[] c0, byte[] code, int d)
        {
            var c = c0;
            var dis = DistL2(x, c, d);
            code[0] = 0;
            for (int j = 1; j < ksub_; j++)
            {
                c = c.Skip(d).ToArray();
                var disij = DistL2(x, c, d);
                if (disij < dis)
                {
                    code[0] = (byte)j;
                    dis = disij;
                }
            }
            return dis;
        }

        public void Estep(float[] x, float[] centroids, byte[] codes, int d, int n)
        {
            for (int i = 0; i < n; i++)
            {
                AssignCentroid(SliceArray(x, i * d), centroids, SliceArray(codes, i), d);
            }
        }

        public void Mstep(float[] x0, float[] centroids, byte[] codes, int d, int n)
        {
            var nelts = Enumerable.Repeat(0, ksub_).ToArray();
            Array.Clear(centroids, 0, d * ksub_);

            var x = x0;
            for (int i = 0; i < n; i++)
            {
                var k = codes[i];
                var c1 = SliceArray(centroids, k * d);
                for (int j = 0; j < d; j++)
                {
                    c1[j] += x[j];
                }
                nelts[k]++;
                x = x.Skip(d).ToArray();
            }

            var c = centroids;
            for (int k = 0; k < ksub_; k++)
            {
                var z = (float)nelts[k];
                if (z != 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        c[j] /= z;
                    }
                }
                c = c.Skip(d).ToArray();
            }
            
            for (int k = 0; k < ksub_; k++)
            {
                if (nelts[k] == 0)
                {
                    var m = 0;
                    while (rng.NextDouble() * (n - ksub_) >= nelts[m] - 1)
                    {
                        m = (m + 1) % ksub_;
                    }

                    Array.Copy(
                        sourceArray: centroids,
                        sourceIndex: m * d,
                        destinationArray: centroids,
                        destinationIndex: k * d,
                        length: d);

                    for (int j = 0; j < d; j++)
                    {
                        var sign = (j % 2) * 2 - 1;
                        centroids[k * d + j] += sign * eps_;
                        centroids[m * d + j] -= sign * eps_;
                    }
                    nelts[k] = nelts[m] / 2;
                    nelts[m] -= nelts[k];
                }
            }
        }

        public void Kmeans(float[] x, float[] c, int n, int d)
        {
            var perm = new int[n];

            for (int i = 0; i < n; i++)
            {
                perm[i] = i;
            }

            perm = ShuffleArray(perm);

            for (int i = 0; i < ksub_; i++)
            {
                Array.Copy(
                    sourceArray: x,
                    sourceIndex: perm[i] * d,
                    destinationArray: c,
                    destinationIndex: i * d,
                    length: d);
            }

            var codes = Enumerable.Repeat<byte>(0, n).ToArray();
            for (int i = 0; i < niter_; i++)
            {
                Estep(x, c, codes, d, n);
                Mstep(x, c, codes, d, n);
            }
        }

        public void Train(int n, float[] x)
        {
            if (n < ksub_)
            {
                throw new ArgumentException($"Matrix too small for quantization, must have at least {ksub_} rows");
            }

            var perm = Enumerable.Repeat(0, n).ToArray();
            var d = dsub_;
            var np = Math.Min(n, max_points_);
            var xslice = new List<float>(np * dsub_);

            for (int m = 0; m < nsubq_; m++)
            {
                if (m == nsubq_ - 1)
                {
                    d = lastdsub_;
                }

                if (np != n)
                {
                    perm = ShuffleArray(perm);
                }

                var xslicearr = xslice.ToArray();

                for (int j = 0; j < np; j++)
                {
                    Array.Copy(
                        sourceArray: x,
                        sourceIndex: perm[j] * dim_ + m * dsub_,
                        destinationArray: xslicearr,
                        destinationIndex: j * d,
                        length: d);
                }

                Kmeans(xslicearr, GetCentroids(m, 0), np, d);
            }
        }

        public float MulCode(float[] x, byte[] codes, int t, float alpha)
        {
            float res = 0f;
            var d = dsub_;
            var code = SliceArray(codes, nsubq_ * t);

            for (int m = 0; m < nsubq_; m++)
            {
                var c = GetCentroids(m, code[m]);

                if (m == nsubq_ - 1)
                {
                    d = lastdsub_;
                }

                for (int n = 0; n < d; n++)
                {
                    res += x[m * dsub_ + n] * c[n];
                }
            }

            return res * alpha;
        }

        public void AddCode(float[] x, byte[] codes, int t, float alpha)
        {
            var d = dsub_;
            var code = SliceArray(codes, nsubq_ * t);

            for (int m = 0; m < nsubq_; m++)
            {
                var c = GetCentroids(m, code[m]);

                if (m == nsubq_ - 1)
                {
                    d = lastdsub_;
                }

                for (int n = 0; n < d; n++)
                {
                    x[m * dsub_ + n] += alpha * c[n];
                }
            }
        }

        public void ComputeCode(float[] x, byte[] code)
        {
            var d = dsub_;
            for (int m = 0; m < nsubq_; m++)
            {
                if (m == nsubq_ - 1)
                {
                    d = lastdsub_;
                }

                AssignCentroid(SliceArray(x, m * dsub_), GetCentroids(m, 0), SliceArray(code, m), d);
            }
        }

        public void ComputeCodes(float[] x, byte[] codes, int n)
        {
            for (int i = 0; i < n; i++)
            {
                ComputeCode(SliceArray(x, i * dim_), SliceArray(codes, i * nsubq_));
            }
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(dim_);
            writer.Write(nsubq_);
            writer.Write(dsub_);
            writer.Write(lastdsub_);

            for (int i = 0; i < centroids_.Length; i++)
            {
                writer.Write(centroids_[i]);
            }
        }

        public void Load(BinaryReader reader)
        {
            dim_ = reader.ReadInt32();
            nsubq_ = reader.ReadInt32();
            dsub_ = reader.ReadInt32();
            lastdsub_ = reader.ReadInt32();

            centroids_ = new float[dim_ * ksub_];
            for (int i = 0; i < centroids_.Length; i++)
            {
                centroids_[i] = reader.ReadSingle();
            }
        }

        private float DistL2(float[] x, float[] y, int d)
        {
            var dist = 0f;
            for (int i = 0; i < d; i++)
            {
                var tmp = x[i] - y[i];
                dist += tmp * tmp;
            }
            return dist;
        }
    }
}
