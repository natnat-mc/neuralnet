#include "internal.h"

uint32_t neuralnet_getLayerCount(const neuralnet_t* N) {
	return N->layerCount;
}

uint32_t neuralnet_getNeuronCount(const neuralnet_t* N) {
	uint32_t total=0;
	for(uint32_t i=0; i<N->layerCount; i++) total+=N->layers[i].neuronCount;
	return total;
}

uint64_t neuralnet_getSynapseCount(const neuralnet_t* N) {
	uint64_t total=0;
	for(uint32_t i=1; i<N->layerCount; i++) total+=(N->layers[i-1].neuronCount+1)*N->layers[i].neuronCount;
	return total;
}

uint8_t neuralnet_getActivationFn(const neuralnet_t* N) {
	return N->activationFn;
}

uint32_t neuralnet_getInputCount(const neuralnet_t* N) {
	return N->layers[0].neuronCount;
}

uint32_t neuralnet_getOutputCount(const neuralnet_t* N) {
	return N->layers[N->layerCount-1].neuronCount;
}

uint32_t neuralnet_getLayerNeuronCount(const neuralnet_t* N, const uint32_t layer) {
	return N->layers[layer].neuronCount;
}

uint32_t neuralnet_getLayerSynapseCount(const neuralnet_t* N, const uint32_t layer) {
	if(layer==0) return 0;
	return N->layers[layer-1].neuronCount+1;
}
