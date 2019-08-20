using System.IO;
using Predictions = System.Collections.Generic.List<System.Tuple<float, int>>;

namespace FastText
{
    public class Meter
    {
        struct Metrics
        {
            public long gold;
            public long predicted;
            public long predictedGold;

            public double Precision()
            {
                if (predicted == 0)
                {
                    return double.NaN;
                }

                return predictedGold / (double)predicted;
            }

            public double Recall()
            {
                if (gold == 0)
                {
                    return double.NaN;
                }

                return predictedGold / (double)gold;
            }

            public double F1Score()
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

        public void Log(int[] labels, Predictions predictions)
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

                if (Utils.Contains(labels, prediction.Item2))
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

        public double Precision(int i)
        {
            return labelMetrics_[i].Precision();
        }

        public double Recall(int i)
        {
            return labelMetrics_[i].Recall();
        }

        public double F1Score(int i)
        {
            return labelMetrics_[i].F1Score();
        }

        public double Precision()
        {
            return metrics_.Precision();
        }

        public double Recall()
        {
            return metrics_.Recall();
        }

        public void WriteGeneralMetrics(TextWriter writer, int k)
        {
            writer.WriteLine($"N\t{nexamples_}");
            writer.WriteLine($"P@{k}\t{metrics_.Precision():0.###}");
            writer.WriteLine($"R@{k}\t{metrics_.Recall():0.###}");
        }
    }
}
