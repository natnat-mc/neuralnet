#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "neuralnet.h"

#define TYPE_NAME "pneuralnetwork"

#define parseArgsMethod(fmt, ...) do { \
	PyObject* pNN; \
	if(!PyArg_ParseTuple(args, "O" fmt, &pNN __VA_OPT__(,)  __VA_ARGS__)) return NULL; \
	N=PyCapsule_GetPointer(pNN, TYPE_NAME); \
} while(0)

#define UNUSED(x) (void)(x)

PyObject* pnn_create(PyObject* self, PyObject* args);
PyObject* pnn_getLayerCount(PyObject* self, PyObject* args);
PyObject* pnn_getActivationFn(PyObject* self, PyObject* args);
PyObject* pnn_getInputCount(PyObject* self, PyObject* args);
PyObject* pnn_getOutputCount(PyObject* self, PyObject* args);
PyObject* pnn_getLayerNeuronCount(PyObject* self, PyObject* args);
PyObject* pnn_getLayerSynapseCount(PyObject* self, PyObject* args);
PyObject* pnn_tickFull(PyObject* self, PyObject* args);
PyObject* pnn_getSynapse(PyObject* self, PyObject* args);
PyObject* pnn_setSynapse(PyObject* self, PyObject* args);
PyObject* pnn_loadBuffer(PyObject* self, PyObject* args);
PyObject* pnn_dumpBuffer(PyObject* self, PyObject* args);
