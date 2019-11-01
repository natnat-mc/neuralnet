#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#include "internal.h"

typedef union {
	struct {
		uint8_t u8[4];
	};
	struct {
		uint16_t u16[2];
	};
	uint32_t u32;
} b32_t;

#define write(data, len) do { \
	memcpy(b, (data), (len)); \
	b+=(len); \
} while(0)
#define read(data, len) do { \
	memcpy((data), b, (len)); \
	b+=(len); \
} while(0)

neuralnet_t* neuralnet_loadBuffer(const void* buf) {
	char* b=(char*) buf;
	b32_t d;
	
	// read layer count
	read(&d, 4);
	uint32_t layerCount=ntohl(d.u32);
	
	// read activation function
	read(&d, 4);
	uint8_t activationFn=d.u8[0];
	
	// read neuron counts
	uint32_t* neuronCount=(uint32_t*) malloc(4*layerCount);
	if(!neuronCount) return NULL;
	read(neuronCount, 4*layerCount);
	for(uint32_t i=0; i<layerCount; i++) neuronCount[i]=ntohl(neuronCount[i]);
	
	// allocate neuralnet struct
	neuralnet_t* N=neuralnet__alloc(layerCount, neuronCount);
	if(!N) {
		free(neuronCount);
		return NULL;
	}
	
	// set activation function
	N->activationFn=activationFn;
	
	// do each layer in turn
	for(uint32_t i=0; i<layerCount; i++) {
		layer_t* L=&N->layers[i];
		L->neuronCount=neuronCount[i];
		
		// do each neuron in turn
		if(i) {
			for(uint32_t j=0; j<neuronCount[i]; j++) {
				read(L->neurons[j].synapses, L->neurons[j].synapseCount*4);
			}
		}
	}
	
	// return the loaded network
	free(neuronCount);
	return N;
}

#define orfail() if(len<expectedLen) return 0
int neuralnet_loadBufferCheck(const void* buf, const uint64_t len) {
	if(len<12) return 0; // at least the 1st layer must be present
	if(len%4) return 0; // buffer is 4-aligned
	
	char* b=(char*) buf;
	b32_t d;
	uint64_t expectedLen=12;
	
	// read layer count
	read(&d, 4);
	uint32_t layerCount=ntohl(d.u32);
	expectedLen+=4*(layerCount-1);
	orfail();
	if(layerCount<1) return 0;
	
	// read activation function
	read(&d, 4);
	if(d.u8[0]>=NN_ACTIVATION_FN_COUNT) return 0;
	
	// read first layer
	uint32_t n, p;
	read(&d, 4);
	n=ntohl(d.u32);
	if(n<1) return 0;
	
	// read all other layers
	for(uint32_t i=1; i<layerCount; i++) {
		p=n;
		read(&d, 4);
		n=ntohl(d.u32);
		if(n<1) return 0;
		expectedLen+=4*n*(p+1);
		orfail();
	}
	
	// if everything went well, we can say that we're good
	return 1;
}
#undef orfail

void neuralnet_dumpBuffer(const neuralnet_t* N, void* buf) {
	char* b=(char*) buf;
	b32_t d;
	
	// write layer count
	d.u32=htonl(N->layerCount);
	write(&d, 4);
	
	// write activation function
	d.u32=0;
	d.u8[0]=N->activationFn;
	write(&d, 4);
	
	// write neuron counts
	for(uint32_t i=0; i<N->layerCount; i++) {
		d.u32=htonl(N->layers[i].neuronCount);
		write(&d, 4);
	}
	
	// write synapses
	for(uint32_t i=1; i<N->layerCount; i++) {
		layer_t* L=&N->layers[i];
		layer_t* L1=&N->layers[i-1];
		
		for(uint32_t j=0; j<L->neuronCount; j++) {
			write(L->neurons[j].synapses, 4*(L1->neuronCount+1));
		}
	}
}

uint64_t neuralnet_dumpBufferLength(const neuralnet_t* N) {
	uint64_t len=8;
	len+=4*N->layerCount;
	for(uint32_t i=1; i<N->layerCount; i++) {
		len+=4*N->layers[i].neuronCount*(N->layers[i-1].neuronCount+1);
	}
	return len;
}
