#ifndef CACHEINVALIDATOR_H
#define CACHEINVALIDATOR_H

template <typename... CACHE_HOLDING_TYPES>
class CacheInvalidator {
 public:
  static void invalidateCaches() { internalInvalidateCache<CACHE_HOLDING_TYPES...>(); }

 private:
  CacheInvalidator() = delete;
  ~CacheInvalidator() = delete;

  template <typename CACHE_HOLDING_TYPE>
  static void internalInvalidateCache() {
    CACHE_HOLDING_TYPE::yieldCacheInvalidator()();
  }

  template <typename FIRST_CACHE_HOLDING_TYPE,
            typename SECOND_CACHE_HOLDING_TYPE,
            typename... REMAINING_CACHE_HOLDING_TYPES>
  static void internalInvalidateCache() {
    FIRST_CACHE_HOLDING_TYPE::yieldCacheInvalidator()();
    internalInvalidateCache<SECOND_CACHE_HOLDING_TYPE,
                            REMAINING_CACHE_HOLDING_TYPES...>();
  }
};

#endif
