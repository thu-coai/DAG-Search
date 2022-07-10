#include "Python.h"
struct __pyx_memoryview_obj;
typedef struct {
  struct __pyx_memoryview_obj *memview;
  char *data;
  Py_ssize_t shape[8];
  Py_ssize_t strides[8];
  Py_ssize_t suboffsets[8];
} __Pyx_memviewslice;