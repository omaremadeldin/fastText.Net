using System;
using System.Diagnostics;

namespace fasttext
{
    public class Vector
    {
        protected float[] data_;

        public float[] data => data_;

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

        public long size()
        {
            return data_.Length;
        }

        public void zero()
        {
            Array.Clear(data_, 0, data_.Length);
        }

        public float norm()
        {
            var sum = 0f;
            for (long i = 0; i < size(); i++)
            {
                sum += data_[i] * data_[i];
            }
            return (float)Math.Sqrt(sum);
        }

        public void mul(float a)
        {
            for (long i = 0; i < size(); i++)
            {
                data_[i] *= a;
            }
        }

        public void addVector(Vector source)
        {
            Debug.Assert(size() == source.size());

            for (long i = 0; i < size(); i++)
            {
                data_[i] += source.data_[i];
            }
        }

        public void addVector(Vector source, float s)
        {
            Debug.Assert(size() == source.size());

            for (long i = 0; i < size(); i++)
            {
                data_[i] += s * source.data_[i];
            }
        }

        public void addRow(Matrix A, long i, float a)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < A.size(0));
            Debug.Assert(size() == A.size(1));

            A.addRowToVector(data_, (int)i, a);
        }

        public void addRow(Matrix A, long i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < A.size(0));
            Debug.Assert(size() == A.size(1));

            A.addRowToVector(data_, (int)i);
        }

        public void mul(Matrix A, Vector vec)
        {
            Debug.Assert(A.size(0) == size());
            Debug.Assert(A.size(1) == vec.size());

            for (long i = 0; i < size(); i++)
            {
                data_[i] = A.dotRow(vec.data_, i);
            }
        }

        public long argmax()
        {
            var max = data_[0];
            long argmax = 0;
            for (long i = 1; i < size(); i++)
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
