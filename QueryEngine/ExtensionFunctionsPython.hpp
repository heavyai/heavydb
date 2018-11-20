/*
  This file is generated using mapd_udf.py script. Do not edit!
*/

#ifdef HAVE_PYTHON
#include <map>
#include <iostream>
#include "Python.h"

int64_t get_pyudf_function(const char * storage_key, int64_t func_id) {
  static std::map<int64_t,int64_t> cache;
  std::map<int64_t,int64_t>::iterator it=cache.find(func_id);
  if (it != cache.end())
    return it->second;
  int64_t cfunc_ptr = 0;
  PyObject* main_module = PyImport_AddModule("__main__"); // borrowed ref
  PyObject* main_dict = PyModule_GetDict(main_module);    // borrowed ref
  PyObject* storages = PyDict_GetItemString(main_dict, "storages"); // borrowed ref
  if (storages != NULL) {
    PyObject* pyudf_key = PyUnicode_FromString(storage_key);
    PyObject* storage = PyObject_GetItem(storages, pyudf_key); // this creates remotedict.Storage object
    Py_DECREF(pyudf_key);
    if (storage != NULL) {
      PyObject* py_func_id = PyLong_FromLongLong(func_id);
      if (py_func_id != NULL) {
        PyObject* py_udf = PyObject_GetItem(storage, py_func_id); // this connects to remotedict server
        if (py_udf != NULL) {
          PyObject* py_address = PyObject_GetAttrString(py_udf, "address"); // this calls numba.cfunc and generates machine code
          if (py_address != NULL) {
            cfunc_ptr = PyLong_AsLongLong(py_address);
            cache[func_id] = cfunc_ptr;
            Py_DECREF(py_address);
          }
          Py_DECREF(py_udf);
        }
        Py_DECREF(py_func_id);
      }
      Py_DECREF(storage);
    } else {
      PyErr_SetString(PyExc_KeyError, storage_key);
    }
  } else {
    PyErr_SetString(PyExc_KeyError, "storages is missing in the __main__ namespace");
  }
  if (PyErr_Occurred()) { PyErr_Print(); }
  return cfunc_ptr;
}


typedef double(*d_d_typedef)(double);

EXTENSION_NOINLINE
double Pyudf_d_d(const int64_t a0, const double a1) {
  d_d_typedef func_ptr = (d_d_typedef)get_pyudf_function("pyudf_d_d", a0);
  //std::cout << "In Pyudf_d_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef double(*dd_d_typedef)(double, double);

EXTENSION_NOINLINE
double Pyudf_dd_d(const int64_t a0, const double a1, const double a2) {
  dd_d_typedef func_ptr = (dd_d_typedef)get_pyudf_function("pyudf_dd_d", a0);
  //std::cout << "In Pyudf_dd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef double(*ddd_d_typedef)(double, double, double);

EXTENSION_NOINLINE
double Pyudf_ddd_d(const int64_t a0, const double a1, const double a2, const double a3) {
  ddd_d_typedef func_ptr = (ddd_d_typedef)get_pyudf_function("pyudf_ddd_d", a0);
  //std::cout << "In Pyudf_ddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3);
}

typedef double(*dddd_d_typedef)(double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_dddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4) {
  dddd_d_typedef func_ptr = (dddd_d_typedef)get_pyudf_function("pyudf_dddd_d", a0);
  //std::cout << "In Pyudf_dddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4);
}

typedef double(*ddddd_d_typedef)(double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_ddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5) {
  ddddd_d_typedef func_ptr = (ddddd_d_typedef)get_pyudf_function("pyudf_ddddd_d", a0);
  //std::cout << "In Pyudf_ddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5);
}

typedef double(*dddddd_d_typedef)(double, double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_dddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5, const double a6) {
  dddddd_d_typedef func_ptr = (dddddd_d_typedef)get_pyudf_function("pyudf_dddddd_d", a0);
  //std::cout << "In Pyudf_dddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6);
}

typedef double(*ddddddd_d_typedef)(double, double, double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_ddddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5, const double a6, const double a7) {
  ddddddd_d_typedef func_ptr = (ddddddd_d_typedef)get_pyudf_function("pyudf_ddddddd_d", a0);
  //std::cout << "In Pyudf_ddddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7);
}

typedef double(*dddddddd_d_typedef)(double, double, double, double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_dddddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5, const double a6, const double a7, const double a8) {
  dddddddd_d_typedef func_ptr = (dddddddd_d_typedef)get_pyudf_function("pyudf_dddddddd_d", a0);
  //std::cout << "In Pyudf_dddddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8);
}

typedef double(*ddddddddd_d_typedef)(double, double, double, double, double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_ddddddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5, const double a6, const double a7, const double a8, const double a9) {
  ddddddddd_d_typedef func_ptr = (ddddddddd_d_typedef)get_pyudf_function("pyudf_ddddddddd_d", a0);
  //std::cout << "In Pyudf_ddddddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

typedef double(*dddddddddd_d_typedef)(double, double, double, double, double, double, double, double, double, double);

EXTENSION_NOINLINE
double Pyudf_dddddddddd_d(const int64_t a0, const double a1, const double a2, const double a3, const double a4, const double a5, const double a6, const double a7, const double a8, const double a9, const double a10) {
  dddddddddd_d_typedef func_ptr = (dddddddddd_d_typedef)get_pyudf_function("pyudf_dddddddddd_d", a0);
  //std::cout << "In Pyudf_dddddddddd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

typedef float(*f_f_typedef)(float);

EXTENSION_NOINLINE
float Pyudf_f_f(const int64_t a0, const float a1) {
  f_f_typedef func_ptr = (f_f_typedef)get_pyudf_function("pyudf_f_f", a0);
  //std::cout << "In Pyudf_f_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef float(*ff_f_typedef)(float, float);

EXTENSION_NOINLINE
float Pyudf_ff_f(const int64_t a0, const float a1, const float a2) {
  ff_f_typedef func_ptr = (ff_f_typedef)get_pyudf_function("pyudf_ff_f", a0);
  //std::cout << "In Pyudf_ff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef float(*fff_f_typedef)(float, float, float);

EXTENSION_NOINLINE
float Pyudf_fff_f(const int64_t a0, const float a1, const float a2, const float a3) {
  fff_f_typedef func_ptr = (fff_f_typedef)get_pyudf_function("pyudf_fff_f", a0);
  //std::cout << "In Pyudf_fff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3);
}

typedef float(*ffff_f_typedef)(float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_ffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4) {
  ffff_f_typedef func_ptr = (ffff_f_typedef)get_pyudf_function("pyudf_ffff_f", a0);
  //std::cout << "In Pyudf_ffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4);
}

typedef float(*fffff_f_typedef)(float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_fffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5) {
  fffff_f_typedef func_ptr = (fffff_f_typedef)get_pyudf_function("pyudf_fffff_f", a0);
  //std::cout << "In Pyudf_fffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5);
}

typedef float(*ffffff_f_typedef)(float, float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_ffffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5, const float a6) {
  ffffff_f_typedef func_ptr = (ffffff_f_typedef)get_pyudf_function("pyudf_ffffff_f", a0);
  //std::cout << "In Pyudf_ffffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6);
}

typedef float(*fffffff_f_typedef)(float, float, float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_fffffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5, const float a6, const float a7) {
  fffffff_f_typedef func_ptr = (fffffff_f_typedef)get_pyudf_function("pyudf_fffffff_f", a0);
  //std::cout << "In Pyudf_fffffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7);
}

typedef float(*ffffffff_f_typedef)(float, float, float, float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_ffffffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5, const float a6, const float a7, const float a8) {
  ffffffff_f_typedef func_ptr = (ffffffff_f_typedef)get_pyudf_function("pyudf_ffffffff_f", a0);
  //std::cout << "In Pyudf_ffffffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8);
}

typedef float(*fffffffff_f_typedef)(float, float, float, float, float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_fffffffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5, const float a6, const float a7, const float a8, const float a9) {
  fffffffff_f_typedef func_ptr = (fffffffff_f_typedef)get_pyudf_function("pyudf_fffffffff_f", a0);
  //std::cout << "In Pyudf_fffffffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

typedef float(*ffffffffff_f_typedef)(float, float, float, float, float, float, float, float, float, float);

EXTENSION_NOINLINE
float Pyudf_ffffffffff_f(const int64_t a0, const float a1, const float a2, const float a3, const float a4, const float a5, const float a6, const float a7, const float a8, const float a9, const float a10) {
  ffffffffff_f_typedef func_ptr = (ffffffffff_f_typedef)get_pyudf_function("pyudf_ffffffffff_f", a0);
  //std::cout << "In Pyudf_ffffffffff_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

typedef int64_t(*l_l_typedef)(int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_l_l(const int64_t a0, const int64_t a1) {
  l_l_typedef func_ptr = (l_l_typedef)get_pyudf_function("pyudf_l_l", a0);
  //std::cout << "In Pyudf_l_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef int64_t(*ll_l_typedef)(int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_ll_l(const int64_t a0, const int64_t a1, const int64_t a2) {
  ll_l_typedef func_ptr = (ll_l_typedef)get_pyudf_function("pyudf_ll_l", a0);
  //std::cout << "In Pyudf_ll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*lll_l_typedef)(int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_lll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3) {
  lll_l_typedef func_ptr = (lll_l_typedef)get_pyudf_function("pyudf_lll_l", a0);
  //std::cout << "In Pyudf_lll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3);
}

typedef int64_t(*llll_l_typedef)(int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_llll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4) {
  llll_l_typedef func_ptr = (llll_l_typedef)get_pyudf_function("pyudf_llll_l", a0);
  //std::cout << "In Pyudf_llll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4);
}

typedef int64_t(*lllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_lllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5) {
  lllll_l_typedef func_ptr = (lllll_l_typedef)get_pyudf_function("pyudf_lllll_l", a0);
  //std::cout << "In Pyudf_lllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5);
}

typedef int64_t(*llllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_llllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5, const int64_t a6) {
  llllll_l_typedef func_ptr = (llllll_l_typedef)get_pyudf_function("pyudf_llllll_l", a0);
  //std::cout << "In Pyudf_llllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6);
}

typedef int64_t(*lllllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_lllllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5, const int64_t a6, const int64_t a7) {
  lllllll_l_typedef func_ptr = (lllllll_l_typedef)get_pyudf_function("pyudf_lllllll_l", a0);
  //std::cout << "In Pyudf_lllllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7);
}

typedef int64_t(*llllllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_llllllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5, const int64_t a6, const int64_t a7, const int64_t a8) {
  llllllll_l_typedef func_ptr = (llllllll_l_typedef)get_pyudf_function("pyudf_llllllll_l", a0);
  //std::cout << "In Pyudf_llllllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8);
}

typedef int64_t(*lllllllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_lllllllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5, const int64_t a6, const int64_t a7, const int64_t a8, const int64_t a9) {
  lllllllll_l_typedef func_ptr = (lllllllll_l_typedef)get_pyudf_function("pyudf_lllllllll_l", a0);
  //std::cout << "In Pyudf_lllllllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

typedef int64_t(*llllllllll_l_typedef)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_llllllllll_l(const int64_t a0, const int64_t a1, const int64_t a2, const int64_t a3, const int64_t a4, const int64_t a5, const int64_t a6, const int64_t a7, const int64_t a8, const int64_t a9, const int64_t a10) {
  llllllllll_l_typedef func_ptr = (llllllllll_l_typedef)get_pyudf_function("pyudf_llllllllll_l", a0);
  //std::cout << "In Pyudf_llllllllll_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

typedef int32_t(*i_i_typedef)(int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_i_i(const int64_t a0, const int32_t a1) {
  i_i_typedef func_ptr = (i_i_typedef)get_pyudf_function("pyudf_i_i", a0);
  //std::cout << "In Pyudf_i_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef int32_t(*ii_i_typedef)(int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_ii_i(const int64_t a0, const int32_t a1, const int32_t a2) {
  ii_i_typedef func_ptr = (ii_i_typedef)get_pyudf_function("pyudf_ii_i", a0);
  //std::cout << "In Pyudf_ii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*iii_i_typedef)(int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3) {
  iii_i_typedef func_ptr = (iii_i_typedef)get_pyudf_function("pyudf_iii_i", a0);
  //std::cout << "In Pyudf_iii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3);
}

typedef int32_t(*iiii_i_typedef)(int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4) {
  iiii_i_typedef func_ptr = (iiii_i_typedef)get_pyudf_function("pyudf_iiii_i", a0);
  //std::cout << "In Pyudf_iiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4);
}

typedef int32_t(*iiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5) {
  iiiii_i_typedef func_ptr = (iiiii_i_typedef)get_pyudf_function("pyudf_iiiii_i", a0);
  //std::cout << "In Pyudf_iiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5);
}

typedef int32_t(*iiiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5, const int32_t a6) {
  iiiiii_i_typedef func_ptr = (iiiiii_i_typedef)get_pyudf_function("pyudf_iiiiii_i", a0);
  //std::cout << "In Pyudf_iiiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6);
}

typedef int32_t(*iiiiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5, const int32_t a6, const int32_t a7) {
  iiiiiii_i_typedef func_ptr = (iiiiiii_i_typedef)get_pyudf_function("pyudf_iiiiiii_i", a0);
  //std::cout << "In Pyudf_iiiiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7);
}

typedef int32_t(*iiiiiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiiiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5, const int32_t a6, const int32_t a7, const int32_t a8) {
  iiiiiiii_i_typedef func_ptr = (iiiiiiii_i_typedef)get_pyudf_function("pyudf_iiiiiiii_i", a0);
  //std::cout << "In Pyudf_iiiiiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8);
}

typedef int32_t(*iiiiiiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiiiiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5, const int32_t a6, const int32_t a7, const int32_t a8, const int32_t a9) {
  iiiiiiiii_i_typedef func_ptr = (iiiiiiiii_i_typedef)get_pyudf_function("pyudf_iiiiiiiii_i", a0);
  //std::cout << "In Pyudf_iiiiiiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

typedef int32_t(*iiiiiiiiii_i_typedef)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_iiiiiiiiii_i(const int64_t a0, const int32_t a1, const int32_t a2, const int32_t a3, const int32_t a4, const int32_t a5, const int32_t a6, const int32_t a7, const int32_t a8, const int32_t a9, const int32_t a10) {
  iiiiiiiiii_i_typedef func_ptr = (iiiiiiiiii_i_typedef)get_pyudf_function("pyudf_iiiiiiiiii_i", a0);
  //std::cout << "In Pyudf_iiiiiiiiii_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}

typedef float(*d_f_typedef)(double);

EXTENSION_NOINLINE
float Pyudf_d_f(const int64_t a0, const double a1) {
  d_f_typedef func_ptr = (d_f_typedef)get_pyudf_function("pyudf_d_f", a0);
  //std::cout << "In Pyudf_d_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef double(*f_d_typedef)(float);

EXTENSION_NOINLINE
double Pyudf_f_d(const int64_t a0, const float a1) {
  f_d_typedef func_ptr = (f_d_typedef)get_pyudf_function("pyudf_f_d", a0);
  //std::cout << "In Pyudf_f_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef float(*dd_f_typedef)(double, double);

EXTENSION_NOINLINE
float Pyudf_dd_f(const int64_t a0, const double a1, const double a2) {
  dd_f_typedef func_ptr = (dd_f_typedef)get_pyudf_function("pyudf_dd_f", a0);
  //std::cout << "In Pyudf_dd_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef double(*df_d_typedef)(double, float);

EXTENSION_NOINLINE
double Pyudf_df_d(const int64_t a0, const double a1, const float a2) {
  df_d_typedef func_ptr = (df_d_typedef)get_pyudf_function("pyudf_df_d", a0);
  //std::cout << "In Pyudf_df_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef float(*df_f_typedef)(double, float);

EXTENSION_NOINLINE
float Pyudf_df_f(const int64_t a0, const double a1, const float a2) {
  df_f_typedef func_ptr = (df_f_typedef)get_pyudf_function("pyudf_df_f", a0);
  //std::cout << "In Pyudf_df_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef double(*fd_d_typedef)(float, double);

EXTENSION_NOINLINE
double Pyudf_fd_d(const int64_t a0, const float a1, const double a2) {
  fd_d_typedef func_ptr = (fd_d_typedef)get_pyudf_function("pyudf_fd_d", a0);
  //std::cout << "In Pyudf_fd_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef float(*fd_f_typedef)(float, double);

EXTENSION_NOINLINE
float Pyudf_fd_f(const int64_t a0, const float a1, const double a2) {
  fd_f_typedef func_ptr = (fd_f_typedef)get_pyudf_function("pyudf_fd_f", a0);
  //std::cout << "In Pyudf_fd_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef double(*ff_d_typedef)(float, float);

EXTENSION_NOINLINE
double Pyudf_ff_d(const int64_t a0, const float a1, const float a2) {
  ff_d_typedef func_ptr = (ff_d_typedef)get_pyudf_function("pyudf_ff_d", a0);
  //std::cout << "In Pyudf_ff_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*d_l_typedef)(double);

EXTENSION_NOINLINE
int64_t Pyudf_d_l(const int64_t a0, const double a1) {
  d_l_typedef func_ptr = (d_l_typedef)get_pyudf_function("pyudf_d_l", a0);
  //std::cout << "In Pyudf_d_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef double(*l_d_typedef)(int64_t);

EXTENSION_NOINLINE
double Pyudf_l_d(const int64_t a0, const int64_t a1) {
  l_d_typedef func_ptr = (l_d_typedef)get_pyudf_function("pyudf_l_d", a0);
  //std::cout << "In Pyudf_l_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef int64_t(*dd_l_typedef)(double, double);

EXTENSION_NOINLINE
int64_t Pyudf_dd_l(const int64_t a0, const double a1, const double a2) {
  dd_l_typedef func_ptr = (dd_l_typedef)get_pyudf_function("pyudf_dd_l", a0);
  //std::cout << "In Pyudf_dd_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*dl_d_typedef)(double, int64_t);

EXTENSION_NOINLINE
double Pyudf_dl_d(const int64_t a0, const double a1, const int64_t a2) {
  dl_d_typedef func_ptr = (dl_d_typedef)get_pyudf_function("pyudf_dl_d", a0);
  //std::cout << "In Pyudf_dl_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*dl_l_typedef)(double, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_dl_l(const int64_t a0, const double a1, const int64_t a2) {
  dl_l_typedef func_ptr = (dl_l_typedef)get_pyudf_function("pyudf_dl_l", a0);
  //std::cout << "In Pyudf_dl_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*ld_d_typedef)(int64_t, double);

EXTENSION_NOINLINE
double Pyudf_ld_d(const int64_t a0, const int64_t a1, const double a2) {
  ld_d_typedef func_ptr = (ld_d_typedef)get_pyudf_function("pyudf_ld_d", a0);
  //std::cout << "In Pyudf_ld_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*ld_l_typedef)(int64_t, double);

EXTENSION_NOINLINE
int64_t Pyudf_ld_l(const int64_t a0, const int64_t a1, const double a2) {
  ld_l_typedef func_ptr = (ld_l_typedef)get_pyudf_function("pyudf_ld_l", a0);
  //std::cout << "In Pyudf_ld_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*ll_d_typedef)(int64_t, int64_t);

EXTENSION_NOINLINE
double Pyudf_ll_d(const int64_t a0, const int64_t a1, const int64_t a2) {
  ll_d_typedef func_ptr = (ll_d_typedef)get_pyudf_function("pyudf_ll_d", a0);
  //std::cout << "In Pyudf_ll_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*d_i_typedef)(double);

EXTENSION_NOINLINE
int32_t Pyudf_d_i(const int64_t a0, const double a1) {
  d_i_typedef func_ptr = (d_i_typedef)get_pyudf_function("pyudf_d_i", a0);
  //std::cout << "In Pyudf_d_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef double(*i_d_typedef)(int32_t);

EXTENSION_NOINLINE
double Pyudf_i_d(const int64_t a0, const int32_t a1) {
  i_d_typedef func_ptr = (i_d_typedef)get_pyudf_function("pyudf_i_d", a0);
  //std::cout << "In Pyudf_i_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef int32_t(*dd_i_typedef)(double, double);

EXTENSION_NOINLINE
int32_t Pyudf_dd_i(const int64_t a0, const double a1, const double a2) {
  dd_i_typedef func_ptr = (dd_i_typedef)get_pyudf_function("pyudf_dd_i", a0);
  //std::cout << "In Pyudf_dd_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*di_d_typedef)(double, int32_t);

EXTENSION_NOINLINE
double Pyudf_di_d(const int64_t a0, const double a1, const int32_t a2) {
  di_d_typedef func_ptr = (di_d_typedef)get_pyudf_function("pyudf_di_d", a0);
  //std::cout << "In Pyudf_di_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*di_i_typedef)(double, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_di_i(const int64_t a0, const double a1, const int32_t a2) {
  di_i_typedef func_ptr = (di_i_typedef)get_pyudf_function("pyudf_di_i", a0);
  //std::cout << "In Pyudf_di_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*id_d_typedef)(int32_t, double);

EXTENSION_NOINLINE
double Pyudf_id_d(const int64_t a0, const int32_t a1, const double a2) {
  id_d_typedef func_ptr = (id_d_typedef)get_pyudf_function("pyudf_id_d", a0);
  //std::cout << "In Pyudf_id_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*id_i_typedef)(int32_t, double);

EXTENSION_NOINLINE
int32_t Pyudf_id_i(const int64_t a0, const int32_t a1, const double a2) {
  id_i_typedef func_ptr = (id_i_typedef)get_pyudf_function("pyudf_id_i", a0);
  //std::cout << "In Pyudf_id_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef double(*ii_d_typedef)(int32_t, int32_t);

EXTENSION_NOINLINE
double Pyudf_ii_d(const int64_t a0, const int32_t a1, const int32_t a2) {
  ii_d_typedef func_ptr = (ii_d_typedef)get_pyudf_function("pyudf_ii_d", a0);
  //std::cout << "In Pyudf_ii_d: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef float(*l_f_typedef)(int64_t);

EXTENSION_NOINLINE
float Pyudf_l_f(const int64_t a0, const int64_t a1) {
  l_f_typedef func_ptr = (l_f_typedef)get_pyudf_function("pyudf_l_f", a0);
  //std::cout << "In Pyudf_l_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef int64_t(*f_l_typedef)(float);

EXTENSION_NOINLINE
int64_t Pyudf_f_l(const int64_t a0, const float a1) {
  f_l_typedef func_ptr = (f_l_typedef)get_pyudf_function("pyudf_f_l", a0);
  //std::cout << "In Pyudf_f_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef float(*ll_f_typedef)(int64_t, int64_t);

EXTENSION_NOINLINE
float Pyudf_ll_f(const int64_t a0, const int64_t a1, const int64_t a2) {
  ll_f_typedef func_ptr = (ll_f_typedef)get_pyudf_function("pyudf_ll_f", a0);
  //std::cout << "In Pyudf_ll_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*lf_l_typedef)(int64_t, float);

EXTENSION_NOINLINE
int64_t Pyudf_lf_l(const int64_t a0, const int64_t a1, const float a2) {
  lf_l_typedef func_ptr = (lf_l_typedef)get_pyudf_function("pyudf_lf_l", a0);
  //std::cout << "In Pyudf_lf_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef float(*lf_f_typedef)(int64_t, float);

EXTENSION_NOINLINE
float Pyudf_lf_f(const int64_t a0, const int64_t a1, const float a2) {
  lf_f_typedef func_ptr = (lf_f_typedef)get_pyudf_function("pyudf_lf_f", a0);
  //std::cout << "In Pyudf_lf_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*fl_l_typedef)(float, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_fl_l(const int64_t a0, const float a1, const int64_t a2) {
  fl_l_typedef func_ptr = (fl_l_typedef)get_pyudf_function("pyudf_fl_l", a0);
  //std::cout << "In Pyudf_fl_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef float(*fl_f_typedef)(float, int64_t);

EXTENSION_NOINLINE
float Pyudf_fl_f(const int64_t a0, const float a1, const int64_t a2) {
  fl_f_typedef func_ptr = (fl_f_typedef)get_pyudf_function("pyudf_fl_f", a0);
  //std::cout << "In Pyudf_fl_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*ff_l_typedef)(float, float);

EXTENSION_NOINLINE
int64_t Pyudf_ff_l(const int64_t a0, const float a1, const float a2) {
  ff_l_typedef func_ptr = (ff_l_typedef)get_pyudf_function("pyudf_ff_l", a0);
  //std::cout << "In Pyudf_ff_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*l_i_typedef)(int64_t);

EXTENSION_NOINLINE
int32_t Pyudf_l_i(const int64_t a0, const int64_t a1) {
  l_i_typedef func_ptr = (l_i_typedef)get_pyudf_function("pyudf_l_i", a0);
  //std::cout << "In Pyudf_l_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef int64_t(*i_l_typedef)(int32_t);

EXTENSION_NOINLINE
int64_t Pyudf_i_l(const int64_t a0, const int32_t a1) {
  i_l_typedef func_ptr = (i_l_typedef)get_pyudf_function("pyudf_i_l", a0);
  //std::cout << "In Pyudf_i_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef int32_t(*ll_i_typedef)(int64_t, int64_t);

EXTENSION_NOINLINE
int32_t Pyudf_ll_i(const int64_t a0, const int64_t a1, const int64_t a2) {
  ll_i_typedef func_ptr = (ll_i_typedef)get_pyudf_function("pyudf_ll_i", a0);
  //std::cout << "In Pyudf_ll_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*li_l_typedef)(int64_t, int32_t);

EXTENSION_NOINLINE
int64_t Pyudf_li_l(const int64_t a0, const int64_t a1, const int32_t a2) {
  li_l_typedef func_ptr = (li_l_typedef)get_pyudf_function("pyudf_li_l", a0);
  //std::cout << "In Pyudf_li_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*li_i_typedef)(int64_t, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_li_i(const int64_t a0, const int64_t a1, const int32_t a2) {
  li_i_typedef func_ptr = (li_i_typedef)get_pyudf_function("pyudf_li_i", a0);
  //std::cout << "In Pyudf_li_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*il_l_typedef)(int32_t, int64_t);

EXTENSION_NOINLINE
int64_t Pyudf_il_l(const int64_t a0, const int32_t a1, const int64_t a2) {
  il_l_typedef func_ptr = (il_l_typedef)get_pyudf_function("pyudf_il_l", a0);
  //std::cout << "In Pyudf_il_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*il_i_typedef)(int32_t, int64_t);

EXTENSION_NOINLINE
int32_t Pyudf_il_i(const int64_t a0, const int32_t a1, const int64_t a2) {
  il_i_typedef func_ptr = (il_i_typedef)get_pyudf_function("pyudf_il_i", a0);
  //std::cout << "In Pyudf_il_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef int64_t(*ii_l_typedef)(int32_t, int32_t);

EXTENSION_NOINLINE
int64_t Pyudf_ii_l(const int64_t a0, const int32_t a1, const int32_t a2) {
  ii_l_typedef func_ptr = (ii_l_typedef)get_pyudf_function("pyudf_ii_l", a0);
  //std::cout << "In Pyudf_ii_l: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef float(*i_f_typedef)(int32_t);

EXTENSION_NOINLINE
float Pyudf_i_f(const int64_t a0, const int32_t a1) {
  i_f_typedef func_ptr = (i_f_typedef)get_pyudf_function("pyudf_i_f", a0);
  //std::cout << "In Pyudf_i_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1);
}

typedef int32_t(*f_i_typedef)(float);

EXTENSION_NOINLINE
int32_t Pyudf_f_i(const int64_t a0, const float a1) {
  f_i_typedef func_ptr = (f_i_typedef)get_pyudf_function("pyudf_f_i", a0);
  //std::cout << "In Pyudf_f_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1);
}

typedef float(*ii_f_typedef)(int32_t, int32_t);

EXTENSION_NOINLINE
float Pyudf_ii_f(const int64_t a0, const int32_t a1, const int32_t a2) {
  ii_f_typedef func_ptr = (ii_f_typedef)get_pyudf_function("pyudf_ii_f", a0);
  //std::cout << "In Pyudf_ii_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*if_i_typedef)(int32_t, float);

EXTENSION_NOINLINE
int32_t Pyudf_if_i(const int64_t a0, const int32_t a1, const float a2) {
  if_i_typedef func_ptr = (if_i_typedef)get_pyudf_function("pyudf_if_i", a0);
  //std::cout << "In Pyudf_if_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef float(*if_f_typedef)(int32_t, float);

EXTENSION_NOINLINE
float Pyudf_if_f(const int64_t a0, const int32_t a1, const float a2) {
  if_f_typedef func_ptr = (if_f_typedef)get_pyudf_function("pyudf_if_f", a0);
  //std::cout << "In Pyudf_if_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*fi_i_typedef)(float, int32_t);

EXTENSION_NOINLINE
int32_t Pyudf_fi_i(const int64_t a0, const float a1, const int32_t a2) {
  fi_i_typedef func_ptr = (fi_i_typedef)get_pyudf_function("pyudf_fi_i", a0);
  //std::cout << "In Pyudf_fi_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}

typedef float(*fi_f_typedef)(float, int32_t);

EXTENSION_NOINLINE
float Pyudf_fi_f(const int64_t a0, const float a1, const int32_t a2) {
  fi_f_typedef func_ptr = (fi_f_typedef)get_pyudf_function("pyudf_fi_f", a0);
  //std::cout << "In Pyudf_fi_f: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return nan("");
  return (*func_ptr)(a1, a2);
}

typedef int32_t(*ff_i_typedef)(float, float);

EXTENSION_NOINLINE
int32_t Pyudf_ff_i(const int64_t a0, const float a1, const float a2) {
  ff_i_typedef func_ptr = (ff_i_typedef)get_pyudf_function("pyudf_ff_i", a0);
  //std::cout << "In Pyudf_ff_i: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return -1;
  return (*func_ptr)(a1, a2);
}


#endif // HAVE_PYTHON
