using System.Diagnostics;
using System.IO;

namespace FastText
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

        public long Size(long dim)
        {
            Debug.Assert(dim == 0 || dim == 1);

            if (dim == 0)
            {
                return m_;
            }

            return n_;
        }

        public abstract float DotRow(float[] vec, long i);

        public abstract void AddVectorToRow(float[] vec, long i, float a);

        public abstract void AddRowToVector(float[] x, int i);

        public abstract void AddRowToVector(float[] x, int i, float a);

        public abstract void Save(BinaryWriter writer);

        public abstract void Load(BinaryReader reader);

        public abstract void Dump(TextWriter writer);
    }
}
