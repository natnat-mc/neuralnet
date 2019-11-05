#include "internal.h"

static PyMethodDef pnn_methods[]={
	{"create",  pnn_create, METH_VARARGS, "Create a neural network"},
	{"getLayerCount", pnn_getLayerCount, METH_VARARGS, "Returns the layer count of a network"},
	{"getActivationFn", pnn_getActivationFn, METH_VARARGS, "Returns the activation function of a network"},
	{"getInputCount", pnn_getInputCount, METH_VARARGS, "Returns the number of input neurons of a network"},
	{"getOutputCount", pnn_getOutputCount, METH_VARARGS, "Returns the number of output neurons of a network"},
	{"getLayerNeuronCount", pnn_getLayerNeuronCount, METH_VARARGS, "Returns the number of neurons in a layer"},
	{"getLayerSynapseCount", pnn_getLayerSynapseCount, METH_VARARGS, "Returns the number of synapses for each neuron in a layer"},
	{"tickFull", pnn_tickFull, METH_VARARGS, "Ticks a neural network with a tuple of inputs"},
	{"getSynapse", pnn_getSynapse, METH_VARARGS, "Returns the coefficient of a single synapse"},
	{"setSynapse", pnn_setSynapse, METH_VARARGS, "Sets the coefficient of a single synapse"},
	{"loadBuffer", pnn_loadBuffer, METH_VARARGS, "Loads a network from a buffer"},
	{"dumpBuffer", pnn_dumpBuffer, METH_VARARGS, "Dumps a network to a buffer"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef pnn_moduleDef={
	PyModuleDef_HEAD_INIT,
	"pneuralnet",
	"A neural network library",
	-1,
	pnn_methods
};

PyMODINIT_FUNC PyInit_pneuralnet(void) {
	PyObject *m=PyModule_Create(&pnn_moduleDef);
	if(!m) return NULL;
	return m;
}
