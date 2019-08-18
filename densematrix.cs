using System;
using System.Diagnostics;
using System.IO;

namespace fasttext
{
    public class DenseMatrix : Matrix
    {
        protected float[] data_;

        public long rows => m_;

        public long cols => n_;

        public float[] data => data_;

        public DenseMatrix() : this(0, 0)
        {
        }

        public DenseMatrix(long m, long n) : base(m, n)
        {
            data_ = new float[m * n];
        }

        public DenseMatrix(DenseMatrix other) :
            this(other.m_, other.n_)
        {
            Array.Copy(other.data_, data_, other.data_.Length);
        }

        public void zero()
        {
            Array.Clear(data_, 0, data_.Length);
        }

        public void uniform(float a)
        {
            var rng = new Random(1);

            for (long i = 0; i < (m_ * n_); i++)
            {
                data_[i] = (float)(rng.NextDouble() * (a - (-a)) + (-a));
            }
        }

        public float at(long i, long j)
        {
            Debug.Assert(i * n_ + j < data_.Length);

            return data_[i * n_ + j];
        }

        public float this[long i, long j]
        {
            get
            {
                return data_[i * n_ + j];
            }
            set
            {
                data_[i * n_ + j] = value;
            }
        }

        public void multiplyRow(float[] nums, long ib = 0, long ie = -1)
        {
            if (ie == -1)
            {
                ie = m_;
            }

            Debug.Assert(ie <= nums.Length);

            for (var i = ib; i < ie; i++)
            {
                var n = nums[i - ib];
                if (n != 0)
                {
                    for (var j = 0; j < n_; j++)
                    {
                        data_[i * n_ + j] *= n;
                    }
                }
            }
        }

        public void divideRow(float[] denoms, long ib = 0, long ie = -1)
        {
            if (ie == -1)
            {
                ie = m_;
            }

            Debug.Assert(ie <= denoms.Length);

            for (var i = ib; i < ie; i++)
            {
                var n = denoms[i - ib];
                if (n != 0)
                {
                    for (var j = 0; j < n_; j++)
                    {
                        data_[i * n_ + j] /= n;
                    }
                }
            }
        }

        public float l2NormRow(long i)
        {
            float norm = 0;
            for (var j = 0; j < n_; j++)
            {
                norm += at(i, j) * at(i, j);
            }
            if (float.IsNaN(norm))
            {
                throw new Exception("Encountered NaN.");
            }

            return (float)Math.Sqrt(norm);
        }

        public void l2NormRow(float[] norms)
        {
            Debug.Assert(norms.Length == m_);

            for (var i = 0; i < m_; i++)
            {
                norms[i] = l2NormRow(i);
            }
        }

        public override float dotRow(float[] vec, long i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < m_);
            Debug.Assert(vec.Length == n_);

            float d = 0;
            for (long j = 0; j < n_; j++)
            {
                d += at(i, j) * vec[j];
            }

            if (float.IsNaN(d))
            {
                throw new Exception("Encountered NaN.");
            }

            return d;
        }

        public override void addVectorToRow(float[] vec, long i, float a)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < m_);
            Debug.Assert(vec.Length == n_);

            for (long j = 0; j < n_; j++)
            {
                data_[i * n_ + j] += a * vec[j];
            }
        }

        public override void addRowToVector(float[] x, int i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < size(0));
            Debug.Assert(x.Length == size(1));

            for (long j = 0; j < n_; j++)
            {
                x[j] += at(i, j);
            }
        }

        public override void addRowToVector(float[] x, int i, float a)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < size(0));
            Debug.Assert(x.Length == size(1));

            for (long j = 0; j < n_; j++)
            {
                x[j] += a * at(i, j);
            }
        }

        public override void save(BinaryWriter writer)
        {
            writer.Write(m_);
            writer.Write(n_);

            for (int i = 0; i < m_ * n_; i++)
            {
                writer.Write(data_[i]);
            }
        }

        public override void load(BinaryReader reader)
        {
            m_ = reader.ReadInt64();
            n_ = reader.ReadInt64();
            data_ = new float[m_ * n_];

            for (int i = 0; i < m_ * n_; i++)
            {
                data_[i] = reader.ReadSingle();
            }
        }

        public override void dump(TextWriter writer)
        {
            writer.WriteLine($"{m_} {n_}");
            for (int i = 0; i < m_; i++)
            {
                for (int j = 0; j < n_; j++)
                {
                    if (j > 0)
                    {
                        writer.Write(" ");
                    }

                    writer.Write(at(i, j));
                }
                writer.Write(Environment.NewLine);
            }
        }
    }
}
