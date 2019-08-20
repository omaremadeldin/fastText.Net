using System;
using System.Diagnostics;

namespace FastText
{
    public class Vector
    {
        protected float[] data_;

        public float[] Data => data_;

        public float this[long key]
        {
            get
            {
                return data_[key];
            }
            set
            {
                data_[key] = value;
            }
        }

        public Vector(long m)
        {
            data_ = new float[m];
        }

        public long Size()
        {
            return data_.Length;
        }

        public void Zero()
        {
            Array.Clear(data_, 0, data_.Length);
        }

        public float Norm()
        {
            var sum = 0f;
            for (long i = 0; i < Size(); i++)
            {
                sum += data_[i] * data_[i];
            }
            return (float)Math.Sqrt(sum);
        }

        public void mul(float a)
        {
            for (long i = 0; i < Size(); i++)
            {
                data_[i] *= a;
            }
        }

        public void AddVector(Vector source)
        {
            Debug.Assert(Size() == source.Size());

            for (long i = 0; i < Size(); i++)
            {
                data_[i] += source.data_[i];
            }
        }

        public void AddVector(Vector source, float s)
        {
            Debug.Assert(Size() == source.Size());

            for (long i = 0; i < Size(); i++)
            {
                data_[i] += s * source.data_[i];
            }
        }

        public void AddRow(Matrix A, long i, float a)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < A.Size(0));
            Debug.Assert(Size() == A.Size(1));

            A.AddRowToVector(data_, (int)i, a);
        }

        public void AddRow(Matrix A, long i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < A.Size(0));
            Debug.Assert(Size() == A.Size(1));

            A.AddRowToVector(data_, (int)i);
        }

        public void Mul(Matrix A, Vector vec)
        {
            Debug.Assert(A.Size(0) == Size());
            Debug.Assert(A.Size(1) == vec.Size());

            for (long i = 0; i < Size(); i++)
            {
                data_[i] = A.DotRow(vec.data_, i);
            }
        }

        public long ArgMax()
        {
            var max = data_[0];
            long argmax = 0;
            for (long i = 1; i < Size(); i++)
            {
                if (data_[i] > max)
                {
                    max = data_[i];
                    argmax = i;
                }
            }
            return argmax;
        }

        public override string ToString()
        {
            var result = string.Empty;

            for (long j = 0; j < data_.Length; j++)
            {
                result += $"{data_[j]:0.#####} ";
            }

            return result;
        }
    }
}
