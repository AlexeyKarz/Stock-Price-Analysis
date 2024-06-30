from datetime import datetime, timedelta
import os
import pickle
from modules.logger import logger


def check_and_clear_cache():
    """
    Check if the cache is older than two days and clear it.
    """
    last_clear = None

    # Check when the cache was last cleared
    if os.path.exists('last_cache_clear.txt'):
        with open('last_cache_clear.txt', 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    last_clear = datetime.fromisoformat(content)
                except ValueError:
                    pass

    # Clear the cache if it's older than two days
    if not last_clear or (datetime.now() - last_clear).days >= 2:
        logger.info("Clearing cache...")
        for filename in os.listdir('cache'):
            file_path = os.path.join('cache', filename)
            os.remove(file_path)
        with open('last_cache_clear.txt', 'w') as f:
            f.write(datetime.now().isoformat())
        logger.info("Cache cleared.")


class Cache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        logger.debug("Cache directory created at: {}".format(cache_dir))

    def get_filename(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol}.pkl")

    def save_data(self, symbol, data):
        filename = self.get_filename(symbol)
        with open(filename, 'wb') as f:
            pickle.dump({
                'date': datetime.now(),
                'data': data
            }, f)
        logger.debug("Data cached for symbol: {}".format(symbol))

    def load_data(self, symbol):
        filename = self.get_filename(symbol)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                cached = pickle.load(f)
                # Check if the cache is still valid, let's say we refresh it every day
                if cached['date'].date() == datetime.now().date():
                    logger.info("Cached data found for symbol: {}".format(symbol))
                    return cached['data']
                else:
                    # Delete old data if it's older than one day
                    os.remove(filename)
        logger.info("No cached data found for symbol: {}".format(symbol))
        return None

    def clear_cache(self):
        """ Clears all cached files if they are older than one day. """
        logger.info("Starting clearing cache...")
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            with open(file_path, 'rb') as f:
                cached = pickle.load(f)
                if (datetime.now() - cached['date']) >= timedelta(days=1):
                    os.remove(file_path)
                    logger.info("Removed cached file: {}".format(filename))


# Create a cache instance to use in the Stock class
cache = Cache()
