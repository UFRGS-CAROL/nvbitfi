/*
 * flex_grip_error_model.h
 *
 *  Created on: Nov 25, 2020
 *      Author: carol
 */

#ifndef FLEX_GRIP_ERROR_MODEL_H_
#define FLEX_GRIP_ERROR_MODEL_H_

__managed__ float xMins[] = { 1.0728769e-07f, 2.0230031e+00f, 8.1847715e-08f,
		1.3602772e+05f, 3.0000000e+00f, 3.5176080e-02f, 3.4028237e+38f,
		2.0000000e+00f, 1.0238367e-02f, 1.3968560e-09f, 2.6865074e-10f,
		1.3970158e-09f, 6.6699225e-01f, 6.6699225e-01f, 6.6699225e-01f,
		7.5000001e-01f, 6.1141304e-01f, 7.5000001e-01f, 0.0000000e+00f,
		7.0958774e-08f, 0.0000000e+00f };
__managed__ float alphas[] = { 1.0868737e+00f, 1.0568325e+00f, 1.0820710e+00f,
		2.7119400e+01f, 1.0678725e+00f, 1.1896030e+00f, 4.4310700e+05f,
		1.4543958e+00f, 1.1181921e+00f, 1.0846596e+00f, 1.0769672e+00f,
		1.0851440e+00f, 2.3798765e+01f, 2.3798765e+01f, 2.3922783e+01f,
		1.2143508e+08f, 3.4316596e+00f, 1.2143508e+08f, 1.0821200e+00f,
		1.0821160e+00f, 1.0821200e+00f };

/**
 * Functions created to inject a FlexGrip
 * Error model on the instructions
 */

#define SMALL_RANGE 1e-3
#define LARGE_RANGE 1e+3

typedef enum {
	FADD_I = 0, FMUL_I, FFMA_I, IADD_I, IMUL_I, IMAD_I, MUFU_I, RELATIVE_INDEX
} RelativeIndex;

typedef enum {
	SMALL = 0, MEDIUM, LARGE, SIZE_TYPES
} ErrorSize;

__inline__     __device__ RelativeIndex get_relative_index(int opcode) {
	switch (opcode) {
	case FFMA:
		return FFMA_I;
	case FADD:
		return FADD_I;
	case FMUL:
		return FMUL_I;
	case IADD:
		return IADD_I;
	case IMUL:
		return IMUL_I;
	case IMAD:
		return IMAD_I;
	case MUFU:
		return MUFU_I;
	default:
		return FADD_I;
	}
}

__inline__     __device__ ErrorSize get_error_size(float beforeVal) {
	float beforeValAbs = fabsf(beforeVal);

	/*Selects the ErrorSize model */
	//small
	if (beforeValAbs < SMALL_RANGE) {
		return SMALL;
	} else if (beforeValAbs > LARGE_RANGE) {
		return LARGE;
	}

	return MEDIUM;
}

__inline__ __device__
float calc_relative_error(float beforeVal, RelativeIndex rI,
		float randomPowerLawInput) {
	ErrorSize eS = get_error_size(beforeVal);
	int relativeIndex = rI * RELATIVE_INDEX + eS;

	//Define powerlaw
	float xMin = xMins[relativeIndex];
	float alpha = alphas[relativeIndex];
	float relativeError = xMin * powf((1.0f - randomPowerLawInput), (-1.0f / (alpha - 1.0f)));

	float valueModified = beforeVal * relativeError;
	return valueModified;
}

/**
 * Function created to inject a FlexGrip
 * Error model on the instructions
 */
__inline__ __device__
bool flex_grip_error_model(unsigned int &afterVal, unsigned int beforeVal,
		int opcode, float randomSeed) {
	bool errorInjected = false;
	unsigned int valueModifiedInt = beforeVal;
	RelativeIndex rI = get_relative_index(opcode);

	switch (opcode) {
	case FFMA:
	case FADD:
	case FMUL:
	case MUFU: {
		float beforeValFloat = *((float*) (&valueModifiedInt));
		float valueModified = calc_relative_error(beforeValFloat, rI,
				randomSeed);
		valueModifiedInt = *((unsigned int*) &valueModified);
		errorInjected = true;
		break;
	}
	case IADD:
	case IMUL:
	case IMAD: {
		// from unsigned to signed
		int beforeValInt = *((int*) (&valueModifiedInt));
		float beforeFloat(beforeValInt);
		int valueModified = int(
				calc_relative_error(beforeFloat, rI, randomSeed));
		valueModifiedInt = *((unsigned int*) &valueModified);
		errorInjected = true;
		break;
	}
		//TODO: Create for ISET and BRA
	}
	afterVal = valueModifiedInt;

	return errorInjected;
}

#endif /* FLEX_GRIP_ERROR_MODEL_H_ */
