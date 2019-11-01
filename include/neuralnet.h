#ifndef __NEURALNET_H
#define __NEURALNET_H

#include <stdint.h>

//BEGIN constants

// function types
#define NN_ACT_RELU 0
#define NN_ACT_SIGMOID 1
#define NN_ACT_SOFTPLUS 2
#define NN_ACT_TANH 3

#define NN_VALID_FN(n) ((n)<4)

//END constants

//BEGIN public neuralnet structures

typedef struct neuralnet_st neuralnet_t;

//END public neuralnet structures

//BEGIN neuralnet functions


//BEGIN constructor and destructor

/**
 * Creates a new neural network with null coefficients
 * @param layerCount the total ammount of layers for this network (including input and output)
 * @cond layerCount>=2
 * @param neuronCount the ammount of neurons per layer, as an array
 * @cond *neuronCount>=1
 * @param activationFn the activation function for this network
 * @returns the newly created neural network, or NULL on failure
 */
neuralnet_t* neuralnet_create(const uint32_t layerCount, const uint32_t* neuronCount, const uint8_t activationFn);

/**
 * Destroys a neural network and frees its memory
 * @param N the network to destroy
 */
void neuralnet_destroy(neuralnet_t* N);

/**
 * Destroys a neural network and frees its memory without freeing the structure itself
 * @param N the network to destroy
 */
void neuralnet_destroy2(neuralnet_t* N);

//END constructor and destructor

//BEGIN statistics functions

/**
 * Returns the layer count of a neural network
 * @param N the network to work on
 * @returns the layer count
 */
uint32_t neuralnet_getLayerCount(const neuralnet_t* N);

/**
 * Returns the total neuron count of a neural network
 * @param N the network to work on
 * @returns the total neuron count
 */
uint32_t neuralnet_getNeuronCount(const neuralnet_t* N);

/**
 * Returns the total synapse count of a neural network
 * @param N the network to work on
 * @returns the total synapse count
 */
uint64_t neuralnet_getSynapseCount(const neuralnet_t* N);

/**
 * Returns the activation function of a neural network
 * @param N the network to work on
 * @returns the activation function constant
 */
uint8_t neuralnet_getActivationFn(const neuralnet_t* N);

/**
 * Returns the number of input neurons of a neural network
 * @param N the network to work on
 * @returns the number of input neurons of the network
 */
uint32_t neuralnet_getInputCount(const neuralnet_t* N);

/**
 * Returns the number of output neurons of a neural network
 * @param N the network to work on
 * @returns the number of output neurons of the network
 */
uint32_t neuralnet_getOutputCount(const neuralnet_t* N);

/**
 * Returns the number of neurons of a layer
 * @param N the network to work on
 * @param layer the layer to work on
 * @returns the number of neurons in the layer
 */
uint32_t neuralnet_getLayerNeuronCount(const neuralnet_t* N, const uint32_t layer);

/**
 * Returns the number of synapse for the neurons of a layer
 * @param N the network to work on
 * @param layer the layer to work on
 * @returns the number of synapses for the neurons in the layer
 */
uint32_t neuralnet_getLayerSynapseCount(const neuralnet_t* N, const uint32_t layer);

//END statistics functions

//BEGIN operation functions

/**
 * Ticks a neural network, that is, processes all of its layers in order
 * @param N the network to tick
 */
void neuralnet_tick(neuralnet_t* N);

/**
 * Ticks a neural network using multiple threads
 * @see neuralnet_tick
 * @param N the network to tick
 * @param threadCount the number of threads to use
 * @cond threadCount>=1
 */
void neuralnet_tickParallel(neuralnet_t* N, const int threadCount);

/**
 * Updates inputs, ticks a network and retrieves outputs
 * @see neuralnet_tick
 * @see neuralnet_setInputs
 * @see neuralnet_getOutputs
 * @param N the network to work on
 * @param threadCount the number of threads to use
 * @param inputs the array of inputs
 * @param outputs the array of outputs
 */
void neuralnet_tickFull(neuralnet_t* N, const int threadCount, const float* inputs, float* outputs);

//END operation functions

//BEGIN learning function

/**
 * Does one learning step
 * @param N the network to work on
 * @param inputs the array of inputs
 * @param outputs the array of expected outputs
 * @param sigma the fraction of the distance to cross
 */
void neuralnet_learn(neuralnet_t* N, const float* inputs, const float*, const float sigma);

//END learning function

//BEGIN getters/setters

/**
 * Sets a neural network's inputs
 * @param N the network to update
 * @param inputs the array of inputs
 */
void neuralnet_setInputs(neuralnet_t* N, const float* inputs);

/**
 * Retreives a neural network's outputs
 * @param N the network to retrieve information from
 * @param outputs the array to update
 */
void neuralnet_getOutputs(neuralnet_t* N, float* outputs);

/**
 * Retreives a single synapse coefficient
 * @param N the network to retrieve information from
 * @param layer the layer to work on
 * @param neuron the neuron to work on
 * @param synapse the synapse to retrieve
 * @returns the coefficient of the selected synapse
 */
float neuralnet_getSynapse(const neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const uint32_t synapse);

/**
 * Sets the coefficient of a single synapse
 * @param N the network to update
 * @param layer the layer to work on
 * @param neuron the neuron to work on
 * @param synapse the synapse to update
 * @param value the coefficient to set
 */
void neuralnet_setSynapse(neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const uint32_t synapse, const float value);

/**
 * Retreives the coefficients of all the synapses of a neuron
 * @param N the network to retreive information from
 * @param layer the layer to work on
 * @param neuron the neuron to retreive synapses from
 * @param values the array to fill
 */
void neuralnet_getNeuronSynapses(const neuralnet_t* N, const uint32_t layer, const uint32_t neuron, float* values);

/**
 * Sets the coefficients of all the synapses of a neuron
 * @param N the network to update
 * @param layer the layer to work on
 * @param neuron the neuron to update
 * @param values the array of coefficients
 */
void neuralnet_setNeuronSynapses(neuralnet_t* N, const uint32_t layer, const uint32_t neuron, const float* values);

/**
 * Retreives the coefficients of all the synapses of a layer
 * @param N the network to retreive information from
 * @param layer the layer to retrieve synapses from
 * @param values the array of arrays to fill
 */
void neuralnet_getLayerSynapses(const neuralnet_t* N, const uint32_t layer, float** values);

/**
 * Sets the coefficients of all the synapses of a layer
 * @param N the network to update
 * @param layer the layer to update
 * @param values the array of arrays of coefficients
 */
void neuralnet_setLayerSynapses(neuralnet_t* N, const uint32_t layer, const float** values);

//END getters/setters

//BEGIN I/O functions

/**
 * Loads a neural network from a dumped buffer
 * @param buf the buffer to load the network from
 * @returns the neural network or NULL if memory allocation failed
 * @warning this function is inherently unsafe, it can pretty much crash the program if the buffer is malformed
 */
neuralnet_t* neuralnet_loadBuffer(const void* buf);

/**
 * Dumps a neural network into a buffer
 * @param N the network to dump
 * @param buf the buffer to dump the network into
 * @warning this function is inherently unsafe, it can pretty much crash the program if the buffer isn't long enough
 */
void neuralnet_dumpBuffer(const neuralnet_t* N, void* buf);

/**
 * Checks if a neural network buffer is valid
 * @param buf the buffer to check
 * @param len the length of the buffer
 * @returns 1 if the buffer is valid, 0 otherwise
 * @see neuralnet_loadBuffer
 */
int neuralnet_loadBufferCheck(const void* buf, const uint64_t len);

/**
 * Returns the buffer size required to dump this neural network
 * @param N the network to check
 * @returns the size of the buffer required, in bytes
 * @see neuralnet_dumpBuffer
 */
uint64_t neuralnet_dumpBufferLength(const neuralnet_t* N);

//END I/O functions

//END neuralnet functions

#endif
