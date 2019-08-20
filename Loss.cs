using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Wintellect.PowerCollections;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

namespace FastText
{
    public abstract class Loss
    {
        private const long LOG_TABLE_SIZE = 512;
        private const long MAX_SIGMOID = 8;
        private const long SIGMOID_TABLE_SIZE = 512;

        protected List<float> t_sigmoid_;
        protected List<float> t_log_;
        protected Matrix wo_;

        protected float Log(float x)
        {
            if (x > 1.0)
            {
                return 0f;
            }

            var i = (int)(x * LOG_TABLE_SIZE);
            return t_log_[i];
        }

        protected float Sigmoid(float x)
        {
            if (x < -MAX_SIGMOID)
            {
                return 0f;
            }
            else if (x > MAX_SIGMOID)
            {
                return 1f;
            }
            else
            {
                var i =
                    (int)((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
                return t_sigmoid_[i];
            }
        }

        protected float StdLog(float x)
        {
            return (float)Math.Log(x + 1E-5);
        }

        private void FindKBest(
            int k,
            float threshold,
            Predictions predictions,
            float[] output)
        {
            var heap = new OrderedBag<Tuple<float, int>>(
                predictions,
                new Comparison<Tuple<float, int>>((l, r) =>
                {
                    var b = l.Item1 > r.Item1;
                    return b ? 1 : 0;
                }));

            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] < threshold)
                {
                    continue;
                }

                if (heap.Count == k && StdLog(output[i]) < heap.First().Item1)
                {
                    continue;
                }

                heap.Add(Tuple.Create(StdLog(output[i]), i));

                if (heap.Count > k)
                {
                    heap.RemoveFirst();
                }
            }
        }

        public Loss(Matrix wo)
        {
            wo_ = wo;
            t_sigmoid_ = new List<float>((int)SIGMOID_TABLE_SIZE + 1);
            for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++)
            {
                var x = (float)(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
                t_sigmoid_.Add(1f / (1f + (float)Math.Exp(-x)));
            }

            t_log_ = new List<float>((int)LOG_TABLE_SIZE + 1);
            for (int i = 0; i < LOG_TABLE_SIZE + 1; i++)
            {
                var x = (i + 1E-5f) / LOG_TABLE_SIZE;
                t_log_.Add((float)Math.Log(x));
            }
        }

        public virtual void Predict(
            int k,
            float threshold,
            Predictions heap,
            Model.State state)
        {
            ComputeOutput(state);
            FindKBest(k, threshold, heap, state.output.Data);
        }

        public abstract float Forward(
            int[] targets,
            int targetIndex,
            Model.State state,
            float lr,
            bool backprop);

        public abstract void ComputeOutput(Model.State state);
    }

    public abstract class BinaryLogisticLoss : Loss
    {
        public BinaryLogisticLoss(Matrix wo) : base(wo)
        {
        }

        public float BinaryLogistic(
            int target,
            Model.State state,
            bool labelIsPositive,
            float lr,
            bool backprop)
        {
            var score = Sigmoid(wo_.DotRow(state.hidden.Data, target));

            if (backprop)
            {
                var flabelIsPositive = (float)Convert.ToDouble(labelIsPositive);
                var alpha = lr * (flabelIsPositive - score);
                state.grad.AddRow(wo_, target, alpha);
                wo_.AddVectorToRow(state.hidden.Data, target, alpha);
            }

            if (labelIsPositive)
            {
                return -Log(score);
            }
            else
            {
                return -Log(1f - score);
            }
        }

        public override void ComputeOutput(Model.State state)
        {
            Vector output = state.output;
            output.Mul(wo_, state.hidden);
            var osz = output.Size();
            for (int i = 0; i < osz; i++)
            {
                output[i] = Sigmoid(output[i]);
            }
        }
    }

    public class OneVsAllLoss : BinaryLogisticLoss
    {
        public OneVsAllLoss(Matrix wo) : base(wo)
        {
        }

        public override float Forward(
            int[] targets,
            int targetIndex,
            Model.State state,
            float lr,
            bool backprop)
        {
            var loss = 0f;
            var osz = state.output.Size();

            for (int i = 0; i < osz; i++)
            {
                bool isMatch = Utils.Contains(targets, i);
                loss += BinaryLogistic(i, state, isMatch, lr, backprop);
            }

            return loss;
        }
    }

    public class NegativeSamplingLoss : BinaryLogisticLoss
    {
        protected const int NEGATIVE_TABLE_SIZE = 10000000;

        protected int neg_;
        protected List<int> negatives_;

        protected int getNegative(int target, Random rng)
        {
            int negative;
            do
            {
                var uniform = rng.Next(0, negatives_.Count);
                negative = negatives_[uniform];
            } while (target == negative);
            return negative;
        }

        public NegativeSamplingLoss(
            Matrix wo,
            int neg,
            long[] targetCounts) : base(wo)
        {
            neg_ = neg;
            negatives_ = new List<int>();

            var z = 0f;

            for (int i = 0; i < targetCounts.Length; i++)
            {
                z += (float)Math.Pow(targetCounts[i], 0.5);
            }
            for (int i = 0; i < targetCounts.Length; i++)
            {
                var c = (float)Math.Pow(targetCounts[i], 0.5);
                for (int j = 0; j < c * NEGATIVE_TABLE_SIZE / z;
                     j++)
                {
                    negatives_.Add(i);
                }
            }
        }

        public override float Forward(
            int[] targets,
            int targetIndex,
            Model.State state,
            float lr,
            bool backprop)
        {
            Debug.Assert(targetIndex >= 0);
            Debug.Assert(targetIndex < targets.Length);

            var target = targets[targetIndex];
            var loss = BinaryLogistic(target, state, true, lr, backprop);

            for (int n = 0; n < neg_; n++)
            {
                var negativeTarget = getNegative(target, state.rng);
                loss += BinaryLogistic(negativeTarget, state, false, lr, backprop);
            }
            return loss;
        }
    }

    public class HierarchicalSoftmaxLoss : BinaryLogisticLoss
    {
        protected class Node
        {
            public int parent;
            public int left;
            public int right;
            public long count;
            public bool binary;
        }

        protected List<int[]> paths_;
        protected List<bool[]> codes_;
        protected List<Node> tree_;
        protected int osz_;

        protected void BuildTree(long[] counts)
        {
            tree_.Capacity = 2 * osz_ - 1;

            for (int i = 0; i < 2 * osz_ - 1; i++)
            {
                tree_[i].parent = -1;
                tree_[i].left = -1;
                tree_[i].right = -1;
                tree_[i].count = (long)1e15;
                tree_[i].binary = false;
            }

            for (int i = 0; i < osz_; i++)
            {
                tree_[i].count = counts[i];
            }

            var leaf = osz_ - 1;
            var node = osz_;
            for (int i = osz_; i < 2 * osz_ - 1; i++)
            {
                var mini = new int[2];

                for (int j = 0; j < 2; j++)
                {
                    if (leaf >= 0 && tree_[leaf].count < tree_[node].count)
                    {
                        mini[j] = leaf--;
                    }
                    else
                    {
                        mini[j] = node++;
                    }
                }

                tree_[i].left = mini[0];
                tree_[i].right = mini[1];
                tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
                tree_[mini[0]].parent = i;
                tree_[mini[1]].parent = i;
                tree_[mini[1]].binary = true;
            }

            for (int i = 0; i < osz_; i++)
            {
                List<int> path = new List<int>();
                List<bool> code = new List<bool>();
                var j = i;

                while (tree_[j].parent != -1)
                {
                    path.Add(tree_[j].parent - osz_);
                    code.Add(tree_[j].binary);
                    j = tree_[j].parent;
                }
                paths_.Add(path.ToArray());
                codes_.Add(code.ToArray());
            }
        }

        protected void DFS(
            int k,
            float threshold,
            int node,
            float score,
            Predictions predictions,
            float[] hidden)
        {
            var heap = new OrderedBag<Tuple<float, int>>(
                predictions,
                new Comparison<Tuple<float, int>>((l, r) =>
                {
                    var b = l.Item1 > r.Item1;
                    return b ? 1 : 0;
                }));

            if (score < StdLog(threshold))
            {
                return;
            }

            if (heap.Count == k && score < heap.GetFirst().Item1)
            {
                return;
            }

            if (tree_[node].left == -1 && tree_[node].right == -1)
            {
                heap.Add(Tuple.Create(score, node));

                if (heap.Count > k)
                {
                    heap.RemoveFirst();
                }

                return;
            }

            var f = wo_.DotRow(hidden, node - osz_);
            f = 1f / (1 + (float)Math.Exp(-f));

            predictions = heap.ToList();

            DFS(k, threshold, tree_[node].left, score + StdLog(1f - f), predictions, hidden);
            DFS(k, threshold, tree_[node].right, score + StdLog(f), predictions, hidden);
        }

        public HierarchicalSoftmaxLoss(Matrix wo, long[] targetCounts) : base(wo)
        {
            paths_ = new List<int[]>();
            codes_ = new List<bool[]>();
            tree_ = new List<Node>();
            osz_ = targetCounts.Length;

            BuildTree(targetCounts);
        }

        public override float Forward(
            int[] targets,
            int targetIndex,
            Model.State state,
            float lr,
            bool backprop)
        {
            var loss = 0f;
            var target = targets[targetIndex];

            var binaryCode = codes_[target];
            var pathToRoot = paths_[target];

            for (int i = 0; i < pathToRoot.Length; i++)
            {
                loss += BinaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
            }
            return loss;
        }

        public override void Predict(
            int k,
            float threshold,
            Predictions heap,
            Model.State state)
        {
            DFS(k, threshold, 2 * osz_ - 2, 0f, heap, state.hidden.Data);
        }
    }

    public class SoftmaxLoss : Loss
    {
        public SoftmaxLoss(Matrix wo) : base(wo)
        {
        }

        public override void ComputeOutput(Model.State state)
        {
            Vector output = state.output;
            output.Mul(wo_, state.hidden);

            var max = output[0];
            var z = 0f;
            var osz = output.Size();

            for (int i = 0; i < osz; i++)
            {
                max = Math.Max(output[i], max);
            }

            for (int i = 0; i < osz; i++)
            {
                output[i] = (float)Math.Exp(output[i] - max);
                z += output[i];
            }

            for (int i = 0; i < osz; i++)
            {
                output[i] /= z;
            }
        }

        public override float Forward(
            int[] targets,
            int targetIndex,
            Model.State state,
            float lr,
            bool backprop)
        {
            Debug.Assert(targetIndex >= 0);
            Debug.Assert(targetIndex < targets.Length);

            ComputeOutput(state);
            var target = targets[targetIndex];

            if (backprop)
            {
                var osz = wo_.Size(0);
                for (int i = 0; i < osz; i++)
                {
                    var label = (i == target) ? 1f : 0f;
                    var alpha = lr * (label - state.output[i]);
                    state.grad.AddRow(wo_, i, alpha);
                    wo_.AddVectorToRow(state.hidden.Data, i, alpha);
                }
            }
            return -Log(state.output[target]);
        }
    }
}
