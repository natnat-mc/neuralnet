#include <math.h>

#include "internal.h"

float neuralnet__activation_relu(const float a) {
	return a>=0?a:0;
}

float neuralnet__activation_sigmoid(const float a) {
	return 1/(1+expf(-a));
}

float neuralnet__activation_softplus(const float a) {
	return logf(1+expf(a));
}

#undef neuralnet__activation_tanh
float neuralnet__activation_tanh(const float a) {
	return tanhf(a);
}

neuralnet__activation_fn_t neuralnet__activationFunctions[]={
	neuralnet__activation_relu,
	neuralnet__activation_sigmoid,
	neuralnet__activation_softplus,
	neuralnet__activation_tanh
};
