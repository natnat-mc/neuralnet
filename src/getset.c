#include "internal.h"

void neuralnet_setInputs(neuralnet_t* N, const float* inputs) {
	layer_t* L=&N->layers[0];
	for(uint32_t i=0; i<L->neuronCount; i++) {
		L->neurons[i].value=inputs[i];
	}
}

void neuralnet_getOutputs(neuralnet_t* N, float* outputs) {
	layer_t* L=&N->layers[N->layerCount-1];
	for(uint32_t i=0; i<L->neuronCount; i++) {
		outputs[i]=L->neurons[i].value;
	}
}

float neuralnet_getSynapse(const neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const uint32_t synapse) {
	return N->layers[layer].neurons[neuron].synapses[synapse];
}

void neuralnet_setSynapse(neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const uint32_t synapse, const float value) {
	N->layers[layer].neurons[neuron].synapses[synapse]=value;
}

void neuralnet_getNeuronSynapses(const neuralnet_t* N, const uint32_t layer, const uint32_t neuron, float* values) {
	neuron_t* M=&N->layers[layer].neurons[neuron];
	float* synapses=M->synapses;
	for(uint32_t i=0; i<M->synapseCount; i++) values[i]=synapses[i];
}

void neuralnet_setNeuronSynapses(neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const float* values) {
	neuron_t* M=&N->layers[layer].neurons[neuron];
	float* synapses=M->synapses;
	for(uint32_t i=0; i<M->synapseCount; i++) synapses[i]=values[i];
}

void neuralnet_getLayerSynapses(const neuralnet_t* N, const uint32_t layer, float** values) {
	layer_t* L=&N->layers[layer];
	for(uint32_t i=0; i<L->neuronCount; i++) {
		neuron_t* M=&L->neurons[i];
		float* synapses=M->synapses;
		float* v=values[i];
		for(uint32_t j=0; j<M->synapseCount; j++) v[j]=synapses[j];
	}
}

void neuralnet_setLayerSynapses(neuralnet_t* N, const uint32_t layer, const float** values) {
	layer_t* L=&N->layers[layer];
	for(uint32_t i=0; i<L->neuronCount; i++) {
		neuron_t* M=&L->neurons[i];
		float* synapses=M->synapses;
		const float* v=values[i];
		for(uint32_t j=0; j<M->synapseCount; j++) synapses[j]=v[j];
	}
}
