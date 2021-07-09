The h3lib subdirectory contains an extension-function-enabled subset of the v3.7.1 tag of the uber h3 github repo https://github.com/uber/h3/tree/v3.7.1 as of 02/02/2021. The h3lib directory here is a modified mirror of the https://github.com/uber/h3/tree/v3.7.1/src/h3lib directory in the repo.

Organizationally, the objective here was to maintain the directory structure of the original h3 repo as closely as possible for easier comparisons. However, due to various limitations with the type system for extension functions, the entire H3 library could not be converted as written. Thus we targeted a small subset of the original repo for conversion, which consists of the geoToH3() method (https://github.com/uber/h3/blob/v3.7.1/src/h3lib/lib/h3Index.c#L737) and its inverse h3ToGeo() (https://github.com/uber/h3/blob/v3.7.1/src/h3lib/lib/h3Index.c#L837). The converted code herein includes only the code paths touched by those two methods. As such there are files missing compared to the uber/h3 repo, as well as commented out code in many of the converted files. Such missing/commented-out code reflects all the other code paths that are not converted to extension functions.

One thing worth noting about the conversion, again due to various limitations with the extension function's type system, most structs defined in the uber/h3 repo could not be converted directly when used as a function argument or return type, such as GeoCoord, and CoordIJK. Fortunately such structs are simple and consist of elements of only one POD type, and therefore can be flattened into a simple POD array, which the type system can recognize. To allow for easier comparisons with the original uber/h3 repo, construction of these new flattened arrays are done via macros.

For example, this is the original definition of the CoordIJK struct from the uber/h3 repo:

```
// from https://github.com/uber/h3/blob/v3.7.1/src/h3lib/include/coordijk.h#L42
typedef struct {
    int i;  ///< i component
    int j;  ///< j component
    int k;  ///< k component
} CoordIJK;
```

for this struct to be used within the extension function type system, `CoordIJK` is flattened to a 3-element int array and is defined with macros like so:

```
#define I_INDEX 0
#define J_INDEX 1
#define K_INDEX 2
#define CoordIJK(variable_name) int variable_name[3]
```

So original code that looked like this:

```
void _setIJK(CoordIJK* ijk, int i, int j, int k) {
  ijk->i = i;
  ijk->j = j;
  ijk->k = k;
}
```

would now look like the following in an extension function:

```
// NOTE: the bool return is also a workaround as the extension function type system
// does not recognize 'void'.
EXTENSION_INLINE bool _setIJK(CoordIJK(ijk), int i, int j, int k) {
  ijk[I_INDEX] = i;
  ijk[J_INDEX] = j;
  ijk[K_INDEX] = k;
  return true;
}
```

This is an initial hack to align the extension function-converted code with the uber/h3 repo as much as possible for more decipherable comparisons.
