using System;
using System.Diagnostics;
using System.IO;

namespace fasttext
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
            : base(mat.size(0), mat.size(1))
        {
            qnorm_ = qnorm;
            codesize_ = (int)(mat.size(0) * ((mat.size(1) + dsub - 1) / dsub));
            codes_ = new byte[codesize_];
            pq_ = new ProductQuantizer((int)n_, dsub);

            if (qnorm_)
            {
                norm_codes_ = new byte[m_];
                npq_ = new ProductQuantizer(1, 1);
            }

            quantize(mat);
        }

        public void quantizeNorm(float[] norms)
        {
            Debug.Assert(qnorm_);
            Debug.Assert(norms.Length == m_);

            npq_.train((int)m_, norms);
            npq_.compute_codes(norms, norm_codes_, (int)m_);
        }

        public void quantize(DenseMatrix mat)
        {
            if (qnorm_)
            {
                var norms = new float[mat.size(0)];
                mat.l2NormRow(norms);
                mat.divideRow(norms);
                quantizeNorm(norms);
            }

            var data = mat.data;
            pq_.train((int)m_, data);
            pq_.compute_codes(data, codes_, (int)m_);
        }

        public override float dotRow(float[] vec, long i)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < m_);
            Debug.Assert(vec.Length == n_);

            float norm = 1;
            if (qnorm_)
            {
                norm = npq_.get_centroids(0, norm_codes_[i])[0];
            }

            return pq_.mulcode(vec, codes_, (int)i, norm);
        }

        public override void addVectorToRow(float[] vec, long i, float a)
        {
            throw new InvalidOperationException("Operation not permitted on quantized matrices.");
        }

        public override void addRowToVector(float[] x, int i, float a)
        {
            var norm = 1f;
            if (qnorm_)
            {
                norm = npq_.get_centroids(0, norm_codes_[i])[0];
            }
            pq_.addcode(x, codes_, i, a * norm);
        }

        public override void addRowToVector(float[] x, int i)
        {
            var norm = 1f;
            if (qnorm_)
            {
                norm = npq_.get_centroids(0, norm_codes_[i])[0];
            }
            pq_.addcode(x, codes_, i, norm);
        }

        public override void save(BinaryWriter writer)
        {
            writer.Write(qnorm_);
            writer.Write(m_);
            writer.Write(n_);
            writer.Write(codesize_);

            for (int i = 0; i < codesize_; i++)
            {
                writer.Write(codes_[i]);
            }

            pq_.save(writer);

            if (qnorm_)
            {
                for (int i = 0; i < m_; i++)
                {
                    writer.Write(norm_codes_[i]);
                }

                npq_.save(writer);
            }
        }

        public override void load(BinaryReader reader)
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
            pq_.load(reader);

            if (qnorm_)
            {
                norm_codes_ = new byte[m_];

                for (int i = 0; i < m_; i++)
                {
                    norm_codes_[i] = reader.ReadByte();
                }

                npq_ = new ProductQuantizer();
                npq_.load(reader);
            }
        }

        public override void dump(TextWriter writer)
        {
            throw new InvalidOperationException("Operation not permitted on quantized matrices.");
        }
    }
}
