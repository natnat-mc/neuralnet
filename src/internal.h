#include "neuralnet.h"

//BEGIN activation functions

#define NN_ACTIVATION_FN_COUNT 4
float neuralnet__activation_relu(const float a);
float neuralnet__activation_sigmoid(const float a);
float neuralnet__activation_softplus(const float a);
float neuralnet__activation_tanh(const float a);
#define neuralnet__activation_tanh tanhf
#include <math.h>

typedef float (*neuralnet__activation_fn_t)(float);
neuralnet__activation_fn_t neuralnet__activationFunctions[NN_ACTIVATION_FN_COUNT];

//END activation functions

//BEGIN initialization and destruction functions

neuralnet_t* neuralnet__alloc(const uint32_t layerCount, const uint32_t* neuronCount);
neuralnet_t* neuralnet__destroy(neuralnet_t* N, int freeN);

//END initialization and destruction functions

//BEGIN private structure

//TODO optimize the layout
#if 0
typedef struct neuralnet_st {
	uint32_t layerCount;
	uint8_t activationFn;
	uint32_t* neuronCount;
	float** values;
	float*** synapses;
} neuralnet_t;
#endif

// forward declaration
typedef struct layer_st layer_t;
typedef struct neuron_st neuron_t;

// network structure
typedef struct neuralnet_st {
	uint32_t layerCount;
	uint8_t activationFn;
	layer_t* layers;
} neuralnet_t;

// layer structure
typedef struct layer_st {
	uint32_t neuronCount;
	neuron_t* neurons;
} layer_t;

// neuron structure
typedef struct neuron_st {
	uint32_t synapseCount;
	float value;
	float* synapses;
} neuron_t;

//END private structure
