using System.Diagnostics;
using System.IO;

namespace fasttext
{
    public abstract class Matrix
    {
        protected long m_;
        protected long n_;

        public Matrix() : this(0, 0)
        {
        }

        public Matrix(long m, long n)
        {
            m_ = m;
            n_ = n;
        }

        public long size(long dim)
        {
            Debug.Assert(dim == 0 || dim == 1);

            if (dim == 0)
            {
                return m_;
            }

            return n_;
        }

        public abstract float dotRow(float[] vec, long i);

        public abstract void addVectorToRow(float[] vec, long i, float a);

        public abstract void addRowToVector(float[] x, int i);

        public abstract void addRowToVector(float[] x, int i, float a);

        public abstract void save(BinaryWriter writer);

        public abstract void load(BinaryReader reader);

        public abstract void dump(TextWriter writer);
    }
}
