#include "internal.h"

void neuralnet_tick(neuralnet_t* N) {
	neuralnet__activation_fn_t activationFn=neuralnet__activationFunctions[N->activationFn];
	for(uint32_t i=1; i<N->layerCount; i++) {
		layer_t* L=&N->layers[i];
		layer_t* L1=&N->layers[i-1];
		for(uint32_t j=0; j<L->neuronCount; j++) {
			neuron_t* M=&L->neurons[j];
			float value=(float) 0;
			for(uint32_t k=0; k<M->synapseCount; k++) value+=M->synapses[k]*L1->neurons[k].value;
			M->value=activationFn(value);
		}
	}
}

void neuralnet_tickParallel(neuralnet_t* N, const int threadCount) {
	if(threadCount==1) {
		neuralnet_tick(N);
		return;
	}
	//TODO actually do it in parallel
	neuralnet_tick(N);
}

void neuralnet_tickFull(neuralnet_t* N, const int threadCount, const float* inputs, float* outputs) {
	neuralnet_setInputs(N, inputs);
	if(threadCount>1) neuralnet_tickParallel(N, threadCount);
	else neuralnet_tick(N);
	neuralnet_getOutputs(N, outputs);
}
