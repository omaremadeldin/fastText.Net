using System;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

namespace fasttext
{
    public class Model
    {
        public class State
        {
            private float lossValue_;
            private long nexamples_;

            public Vector hidden;
            public Vector output;
            public Vector grad;
            public Random rng;

            public State(int hiddenSize, int outputSize, int seed)
            {
                lossValue_ = 0f;
                nexamples_ = 0;
                hidden = new Vector(hiddenSize);
                output = new Vector(outputSize);
                grad = new Vector(hiddenSize);
                rng = new Random(seed);
            }

            public float getLoss()
            {
                return lossValue_ / nexamples_;
            }

            public void incrementNExamples(float loss)
            {
                lossValue_ += loss;
                nexamples_++;
            }
        }

        protected Matrix wi_;
        protected Matrix wo_;
        protected Loss loss_;
        protected bool normalizeGradient_;

        public const int kUnlimitedPredictions = -1;
        public const int kAllLabelsAsTarget = -1;

        public Model(
            Matrix wi,
            Matrix wo,
            Loss loss,
            bool normalizeGradient)
        {
            wi_ = wi;
            wo_ = wo;
            loss_ = loss;
            normalizeGradient_ = normalizeGradient;
        }

        public void predict(
            int[] input,
            int k,
            float threshold,
            Predictions heap,
            State state)
        {
            if (k == kUnlimitedPredictions)
            {
                k = (int)wo_.size(0); // output size
            }
            else if (k <= 0)
            {
                throw new ArgumentException("k needs to be 1 or higher!");
            }
            heap = new Predictions(k + 1);
            computeHidden(input, state);

            loss_.predict(k, threshold, heap, state);
        }

        public void update(
            int[] input,
            int[] targets,
            int targetIndex,
            float lr,
            State state)
        {
            if (input.Length == 0)
            {
                return;
            }
            computeHidden(input, state);

            var grad = state.grad;
            grad.zero();

            var lossValue = loss_.forward(targets, targetIndex, state, lr, true);
            state.incrementNExamples(lossValue);

            if (normalizeGradient_)
            {
                grad.mul(1f / input.Length);
            }
            for (int i = 0; i < input.Length; i++)
            {
                wi_.addVectorToRow(grad.data, i, 1f);
            }
        }

        public void computeHidden(int[] input, State state)
        {
            var hidden = state.hidden;
            hidden.zero();

            for (int i = 0; i < input.Length; i++)
            {
                hidden.addRow(wi_, input[i]);
            }
            hidden.mul(1f / input.Length);
        }

        public float std_log(float x)
        {
            return (float)Math.Log(x + 1E-5f);
        }
    }
}
