from typing import Any, Dict, Optional, TypeVar, Generic, Callable
import time
import threading
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
T = TypeVar('T')

class TimedCache(Generic[T]):
    """A thread-safe cache with time-based expiration."""
    
    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache with time-to-live in seconds."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[T]:
        """Get item from cache if it exists and hasn't expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry['timestamp'] < self._ttl_seconds:
                    return entry['value']
                else:
                    # Remove expired entry
                    del self._cache[key]
        return None
    
    def put(self, key: str, value: T) -> None:
        """Add item to cache with current timestamp."""
        with self._lock:
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def invalidate(self, key: str) -> None:
        """Remove item from cache if it exists."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()


def timed_lru_cache(maxsize: int = 128, ttl_seconds: int = 600):
    """
    Decorator that creates an LRU cache of results with time-based expiration.
    """
    def decorator(func):
        cache = {}
        lock = threading.RLock()
        
        @lru_cache(maxsize=maxsize)
        def cached_func(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            return func(*args, **kwargs)
        
        def wrapper(*args, **kwargs):
            with lock:
                key = (args, frozenset(kwargs.items()))
                current_time = time.time()
                
                if key in cache:
                    result, timestamp = cache[key]
                    if current_time - timestamp < ttl_seconds:
                        return result
                
                result = func(*args, **kwargs)
                cache[key] = (result, current_time)
                
                # Clean up old entries if cache is too large
                if len(cache) > maxsize:
                    oldest_key = min(cache.items(), key=lambda x: x[1][1])[0]
                    del cache[oldest_key]
                    
                return result
                
        return wrapper
    return decorator


class ModelCache:
    """Cache for loaded ML models to avoid reloading costs."""
    
    def __init__(self, max_size: int = 5):
        """Initialize model cache with a maximum size."""
        self._models = {}
        self._max_size = max_size
        self._usage_count = {}
        self._lock = threading.RLock()
        
    def get(self, model_key: str) -> Optional[Any]:
        """Get a model from cache if it exists."""
        with self._lock:
            if model_key in self._models:
                # Update usage count
                self._usage_count[model_key] += 1
                return self._models[model_key]
        return None
        
    def put(self, model_key: str, model: Any) -> None:
        """Add a model to cache, evicting least used if at capacity."""
        with self._lock:
            # If we're at capacity, remove least used model
            if len(self._models) >= self._max_size and model_key not in self._models:
                # Find least used model
                least_used = min(self._usage_count.items(), key=lambda x: x[1])[0]
                logger.info(f"Evicting model {least_used} from cache")
                del self._models[least_used]
                del self._usage_count[least_used]
            
            # Add new model
            self._models[model_key] = model
            self._usage_count[model_key] = 1
