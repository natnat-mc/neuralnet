#include "internal.h"

void pnn_destruct(PyObject* pN) {
	neuralnet_destroy(PyCapsule_GetPointer(pN, TYPE_NAME));
}

PyObject* pnn_create(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	uint32_t layerCount;
	uint32_t* neuronCount;
	uint8_t activationFn;
	
	// argument parsing
	{
		unsigned int a;
		PyObject* neuronCountTuple;
		PyObject* b;
		unsigned char c;
		
		if(!PyArg_ParseTuple(args, "IOb:Expects an int, a tuple of ints and an int", &a, &neuronCountTuple, &c)) return NULL;
		if(!PyTuple_Check(neuronCountTuple)) {
			PyErr_SetString(PyExc_TypeError, "Expects an int, a tuple of ints and an int");
			return NULL;
		}
		layerCount=(uint32_t) a;
		activationFn=(uint8_t) c;
		
		if(a<1) {
			PyErr_SetString(PyExc_ValueError, "Layer count must be at least one");
			return NULL;
		}
		if(((Py_ssize_t) a)!=PyTuple_Size(neuronCountTuple)) {
			PyErr_SetString(PyExc_ValueError, "Size of neuron count tuple must be same as layer count");
			return NULL;
		}
		if(!NN_VALID_FN(c)) {
			PyErr_SetString(PyExc_ValueError, "Activation function is invalid");
			return NULL;
		}
		
		neuronCount=(uint32_t*) malloc(4*layerCount);
		if(!neuronCount) return PyErr_NoMemory();
		
		for(uint32_t i=0; i<layerCount; i++) {
			b=PyTuple_GetItem(neuronCountTuple, (Py_ssize_t) i);
			Py_ssize_t val=PyNumber_AsSsize_t(b, NULL);
			if(val<0) {
				free(neuronCount);
				return NULL;
			}
			neuronCount[i]=(uint32_t) val;
		}
	}
	
	neuralnet_t* N=neuralnet_create(layerCount, neuronCount, activationFn);
	free(neuronCount);
	if(!N) return PyErr_NoMemory();
	
	return PyCapsule_New(N, TYPE_NAME, pnn_destruct);
}

PyObject* pnn_getLayerCount(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	parseArgsMethod("");
	
	return PyLong_FromLong(neuralnet_getLayerCount(N));
}

PyObject* pnn_getActivationFn(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	parseArgsMethod("");
	
	return PyLong_FromLong(neuralnet_getActivationFn(N));
}

PyObject* pnn_getInputCount(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	parseArgsMethod("");
	
	return PyLong_FromLong(neuralnet_getInputCount(N));
}

PyObject* pnn_getOutputCount(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	parseArgsMethod("");
	
	return PyLong_FromLong(neuralnet_getOutputCount(N));
}

PyObject* pnn_getLayerNeuronCount(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	int layer;
	parseArgsMethod("I", &layer);
	
	if(layer<0 || ((uint32_t) layer)>=neuralnet_getLayerCount(N)) {
		PyErr_SetString(PyExc_ValueError, "Layer out of bounds");
		return NULL;
	}
	
	return PyLong_FromLong(neuralnet_getLayerNeuronCount(N, layer));
}

PyObject* pnn_getLayerSynapseCount(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	int layer;
	parseArgsMethod("I", &layer);
	
	if(layer<0 || ((uint32_t) layer)>=neuralnet_getLayerCount(N)) {
		PyErr_SetString(PyExc_ValueError, "Layer out of bounds");
		return NULL;
	}
	
	return PyLong_FromLong(neuralnet_getLayerSynapseCount(N, layer));
}

PyObject* pnn_tickFull(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	int threadCount;
	PyObject* inputTuple;
	parseArgsMethod("IO", &threadCount, &inputTuple);
	
	// fetch data from the network
	uint32_t inputCount=neuralnet_getInputCount(N);
	uint32_t outputCount=neuralnet_getOutputCount(N);
	
	// validate args
	if(threadCount<1) {
		PyErr_SetString(PyExc_ValueError, "Thread count must be at least 1");
		return NULL;
	}
	if(PyTuple_Size(inputTuple)!=inputCount) {
		PyErr_SetString(PyExc_ValueError, "Wrong number of inputs");
		return NULL;
	}
	
	// convert data from Python to C
	float* inputValues;
	float* outputValues;
	inputValues=calloc(inputCount, sizeof(float));
	outputValues=calloc(outputCount, sizeof(float));
	if(!inputValues || !outputValues) {
		free(inputValues);
		free(outputValues);
		return PyErr_NoMemory();
	}
	for(uint32_t i=0; i<inputCount; i++) {
		PyObject* o=PyTuple_GetItem(inputTuple, i);
		inputValues[i]=PyFloat_AsDouble(o);
		if(PyErr_Occurred()) {
			free(inputValues);
			free(outputValues);
			return NULL;
		}
	}
	
	// in soviet Russia, C functions call you
	neuralnet_tickFull(N, threadCount, inputValues, outputValues);
	
	// convert data back to Python
	PyObject* outputTuple=PyTuple_New(outputCount);
	for(uint32_t i=0; i<outputCount; i++) {
		PyTuple_SetItem(outputTuple, i, PyFloat_FromDouble((double) outputValues[i]));
	}
	
	// free our temp memory
	free(inputValues);
	free(outputValues);
	
	// return our tuple to Python
	return outputTuple;
}

PyObject* pnn_getSynapse(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	int layer, neuron, synapse;
	parseArgsMethod("III", &layer, &neuron, &synapse);
	
	// sanity check
	if(layer<0 || neuron<0 || synapse<0) {
		PyErr_SetString(PyExc_ValueError, "Arguments must be positive");
		return NULL;
	}
	
	// fetch data from network
	uint32_t layerCount=neuralnet_getLayerCount(N);
	if((uint32_t) layer>=layerCount) {
		PyErr_SetString(PyExc_ValueError, "Layer out of bounds");
		return NULL;
	}
	uint32_t neuronCount=neuralnet_getLayerNeuronCount(N, layer);
	if((uint32_t) neuron>=neuronCount) {
		PyErr_SetString(PyExc_ValueError, "Neuron out of bounds");
		return NULL;
	}
	uint32_t synapseCount=neuralnet_getLayerSynapseCount(N, layer);
	if((uint32_t) synapse>=synapseCount) {
		PyErr_SetString(PyExc_ValueError, "Synapse out of bounds");
		return NULL;
	}
	
	return PyFloat_FromDouble((double) neuralnet_getSynapse(N, layer, neuron, synapse));
}

PyObject* pnn_setSynapse(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	int layer, neuron, synapse;
	float value;
	parseArgsMethod("IIIf", &layer, &neuron, &synapse, &value);
	
	// sanity check
	if(layer<0 || neuron<0 || synapse<0) {
		PyErr_SetString(PyExc_ValueError, "Arguments must be positive");
		return NULL;
	}
	
	// fetch data from network
	uint32_t layerCount=neuralnet_getLayerCount(N);
	if((uint32_t) layer>=layerCount) {
		PyErr_SetString(PyExc_ValueError, "Layer out of bounds");
		return NULL;
	}
	uint32_t neuronCount=neuralnet_getLayerNeuronCount(N, layer);
	if((uint32_t) neuron>=neuronCount) {
		PyErr_SetString(PyExc_ValueError, "Neuron out of bounds");
		return NULL;
	}
	uint32_t synapseCount=neuralnet_getLayerSynapseCount(N, layer);
	if((uint32_t) synapse>=synapseCount) {
		PyErr_SetString(PyExc_ValueError, "Synapse out of bounds");
		return NULL;
	}
	
	neuralnet_setSynapse(N, layer, neuron, synapse, value);
	return Py_None;
}

PyObject* pnn_loadBuffer(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	void* buf;
	Py_ssize_t pyLen;
	PyArg_ParseTuple(args, "y#", &buf, &pyLen);
	
	// call the actual functions
	if(!neuralnet_loadBufferCheck(buf, (uint64_t) pyLen)) {
		free(buf);
		PyErr_SetString(PyExc_ValueError, "Invalid buffer");
		return NULL;
	}
	neuralnet_t* N=neuralnet_loadBuffer(buf);
	if(!N) return PyErr_NoMemory();
	
	// pass the network back to Python
	return PyCapsule_New(N, TYPE_NAME, pnn_destruct);
}

PyObject* pnn_dumpBuffer(PyObject* self, PyObject* args) {
	UNUSED(self);
	
	// actual arguments
	neuralnet_t* N;
	parseArgsMethod("");
	
	// fetch data from network
	uint64_t len=neuralnet_dumpBufferLength(N);
	void* buf=malloc(len);
	if(!buf) return PyErr_NoMemory();
	neuralnet_dumpBuffer(N, buf);
	
	// transfer back to Python
	PyObject* bytes=PyBytes_FromStringAndSize(buf, len);
	free(buf);
	return bytes;
}

/* TODO for speed
neuralnet_getNeuronSynapses
neuralnet_setNeuronSynapses
neuralnet_getLayerSynapses
neuralnet_setLayerSynapses
*/

/* TODO for completeness
neuralnet_getNeuronCount
neuralnet_getSynapseCount
*/
