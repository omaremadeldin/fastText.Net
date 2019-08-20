namespace FastText
{
    public static class Utils
    {
        public static bool Contains<T>(T[] container, T value)
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
