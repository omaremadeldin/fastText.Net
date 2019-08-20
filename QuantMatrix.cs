using System;
using System.Diagnostics;
using System.IO;

namespace FastText
{
    public class QuantMatrix : Matrix
    {
        protected ProductQuantizer pq_;
        protected ProductQuantizer npq_;

        protected byte[] codes_;
        protected byte[] norm_codes_;

        protected bool qnorm_;
        protected int codesize_;

        public QuantMatrix() : base()
        {
            qnorm_ = false;
            codesize_ = 0;
        }

        public QuantMatrix(DenseMatrix mat, int dsub, bool qnorm)
            : base(mat.Size(0), mat.Size(1))
        {
            qnorm_ = qnorm;
            codesize_ = (int)(mat.Size(0) * ((mat.Size(1) + dsub - 1) / dsub));
            codes_ = new byte[codesize_];
            pq_ = new ProductQuantizer((int)n_, dsub);

            if (qnorm_)
            {
                norm_codes_ = new byte[m_];
                npq_ = new ProductQuantizer(1, 1);
            }

            Quantize(mat);
        }

        public void QuantizeNorm(float[] norms)
        {
            Debug.Assert(qnorm_);
            Debug.Assert(norms.Length == m_);

            npq_.Train((int)m_, norms);
            npq_.ComputeCodes(norms, norm_codes_, (int)m_);
        }

        public void Quantize(DenseMatrix mat)
        {
            if (qnorm_)
            {
                var norms = new float[mat.Size(0)];
                mat.L2NormRow(norms);
                mat.DivideRow(norms);
                QuantizeNorm(norms);
            }

            var data = mat.data;
            pq_.Train((int)m_, data);
            pq_.ComputeCodes(data, codes_, (int)m_);
        }

        public override float DotRow(float[] vec, long i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < m_);
            Debug.Assert(vec.Length == n_);

            float norm = 1;
            if (qnorm_)
            {
                norm = npq_.GetCentroids(0, norm_codes_[i])[0];
            }

            return pq_.MulCode(vec, codes_, (int)i, norm);
        }

        public override void AddVectorToRow(float[] vec, long i, float a)
        {
            throw new InvalidOperationException("Operation not permitted on quantized matrices.");
        }

        public override void AddRowToVector(float[] x, int i, float a)
        {
            var norm = 1f;
            if (qnorm_)
            {
                norm = npq_.GetCentroids(0, norm_codes_[i])[0];
            }
            pq_.AddCode(x, codes_, i, a * norm);
        }

        public override void AddRowToVector(float[] x, int i)
        {
            var norm = 1f;
            if (qnorm_)
            {
                norm = npq_.GetCentroids(0, norm_codes_[i])[0];
            }
            pq_.AddCode(x, codes_, i, norm);
        }

        public override void Save(BinaryWriter writer)
        {
            writer.Write(qnorm_);
            writer.Write(m_);
            writer.Write(n_);
            writer.Write(codesize_);

            for (int i = 0; i < codesize_; i++)
            {
                writer.Write(codes_[i]);
            }

            pq_.Save(writer);

            if (qnorm_)
            {
                for (int i = 0; i < m_; i++)
                {
                    writer.Write(norm_codes_[i]);
                }

                npq_.Save(writer);
            }
        }

        public override void Load(BinaryReader reader)
        {
            qnorm_ = reader.ReadBoolean();
            m_ = reader.ReadInt64();
            n_ = reader.ReadInt64();
            codesize_ = reader.ReadInt32();

            codes_ = new byte[codesize_];
            for(int i = 0; i < codesize_; i++)
            {
                codes_[i] = reader.ReadByte();
            }

            pq_ = new ProductQuantizer();
            pq_.Load(reader);

            if (qnorm_)
            {
                norm_codes_ = new byte[m_];

                for (int i = 0; i < m_; i++)
                {
                    norm_codes_[i] = reader.ReadByte();
                }

                npq_ = new ProductQuantizer();
                npq_.Load(reader);
            }
        }

        public override void Dump(TextWriter writer)
        {
            throw new InvalidOperationException("Operation not permitted on quantized matrices.");
        }
    }
}
