#include <stdlib.h>

#include "internal.h"

#define orfail(a) if(!(a)) return neuralnet__destroy(N, 0)
neuralnet_t* neuralnet__alloc(const uint32_t layerCount, const uint32_t* neuronCount) {
	neuralnet_t* N=calloc(1, sizeof(neuralnet_t));
	orfail(N);
	
	N->layerCount=layerCount;
	
	N->layers=calloc(layerCount, sizeof(layer_t));
	orfail(N->layers);
	
	for(uint32_t i=0; i<layerCount; i++) {
		layer_t* L=&N->layers[i];
		L->neuronCount=neuronCount[i];
		
		L->neurons=calloc(1+neuronCount[i], sizeof(neuron_t));
		orfail(L->neurons);
		
		for(uint32_t j=0; j<neuronCount[i]; j++) {
			neuron_t* M=&L->neurons[j];
			
			if(i) {
				M->synapseCount=1+neuronCount[i-1];
				
				M->synapses=calloc(M->synapseCount, sizeof(float));
				orfail(M->synapses);
			} else {
				M->synapseCount=0;
			}
		}
		
		L->neurons[neuronCount[i]].value=(float) 1;
	}
	
	return N;
}
#undef orfail

neuralnet_t* neuralnet__destroy(neuralnet_t* N, int freeN) {
	if(N) {
		if(N->layers) {
			for(uint32_t i=0; i<N->layerCount; i++) {
				layer_t* L=&(N->layers[i]);
				if(L->neurons) {
					for(uint32_t j=0; j<L->neuronCount; j++) {
						neuron_t* n=&(L->neurons[i]);
						if(n->synapses) {
							free(n->synapses);
							n->synapses=NULL;
						}
					}
					free(L->neurons);
					L->neurons=NULL;
				}
			}
			free(N->layers);
			N->layers=NULL;
		}
		if(freeN) free(N);
	}
	return NULL;
}

neuralnet_t* neuralnet_create(const uint32_t layerCount, const uint32_t* neuronCount, const uint8_t activationFn) {
	neuralnet_t* N=neuralnet__alloc(layerCount, neuronCount);
	if(!N) return NULL;
	
	N->activationFn=activationFn;
	return N;
}

void neuralnet_destroy(neuralnet_t* N) {
	neuralnet__destroy(N, 1);
}

void neuralnet_destroy2(neuralnet_t* N) {
	neuralnet__destroy(N, 0);
}
