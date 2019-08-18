using System.Collections.Generic;

namespace fasttext
{
    public static class Utils
    {
        public static bool contains<T>(T[] container, T value)
        {
            for (int i = 0; i < container.Length; i++)
            {
                if (container[i].Equals(value))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
