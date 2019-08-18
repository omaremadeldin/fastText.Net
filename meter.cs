
using System.IO;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

namespace fasttext
{
    public class Meter
    {
        struct Metrics
        {
            public long gold;
            public long predicted;
            public long predictedGold;

            public double precision()
            {
                if (predicted == 0)
                {
                    return double.NaN;
                }

                return predictedGold / (double)predicted;
            }

            public double recall()
            {
                if (gold == 0)
                {
                    return double.NaN;
                }

                return predictedGold / (double)gold;
            }

            public double f1Score()
            {
                if (predicted + gold == 0)
                {
                    return double.NaN;
                }

                return 2 * predictedGold / (double)(predicted + gold);
            }
        }

        private Metrics metrics_ = new Metrics();
        private long nexamples_;
        private System.Collections.Generic.Dictionary<int, Metrics> labelMetrics_ = new System.Collections.Generic.Dictionary<int, Metrics>();

        public long nexamples => nexamples_;

        public void log(int[] labels, Predictions predictions)
        {
            nexamples_++;
            metrics_.gold += labels.Length;
            metrics_.predicted += predictions.Count;

            for (int i = 0; i < predictions.Count; i++)
            {
                var prediction = predictions[i];
                var metrics = labelMetrics_[prediction.Item2];
                metrics.predicted++;
                labelMetrics_[prediction.Item2] = metrics;

                if (Utils.contains(labels, prediction.Item2))
                {
                    metrics = labelMetrics_[prediction.Item2];
                    metrics.predictedGold++;
                    labelMetrics_[prediction.Item2] = metrics;

                    metrics_.predictedGold++;
                }
            }

            for (int i = 0; i < labels.Length; i++)
            {
                var label = labels[i];
                var metrics = labelMetrics_[label];
                metrics.gold++;
                labelMetrics_[label] = metrics;
            }
        }

        public double precision(int i)
        {
            return labelMetrics_[i].precision();
        }

        public double recall(int i)
        {
            return labelMetrics_[i].recall();
        }

        public double f1Score(int i)
        {
            return labelMetrics_[i].f1Score();
        }

        public double precision()
        {
            return metrics_.precision();
        }

        public double recall()
        {
            return metrics_.recall();
        }

        public void writeGeneralMetrics(TextWriter writer, int k)
        {
            writer.WriteLine($"N\t{nexamples_}");
            writer.WriteLine($"P@{k}\t{metrics_.precision():0.###}");
            writer.WriteLine($"R@{k}\t{metrics_.recall():0.###}");
        }
    }
}
