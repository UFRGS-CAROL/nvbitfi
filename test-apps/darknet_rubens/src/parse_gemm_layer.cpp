/*
 * parse_gemm_layer.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: fernando
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <random>

#include "cuda_templates.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define DEBUG 0

typedef enum {
	SIMULATE_SCHEDULER_FAULT, INVALID_OPTION
} LayerOperationType;

#ifndef FLEX_GRIP_ANALYSIS
#define FLEX_GRIP_ANALYSIS 1
#endif

LayerOperationType operation_type = SIMULATE_SCHEDULER_FAULT;
auto reset_counters_var = false;

#define MAX_FLOAT_THRESHOLD 1.0e-5f

/**
 * Type that represents the fault description
 */

struct FaultDescriptor {
	float min_relative_error, max_relative_error;
	int layer_i;
	std::string geometry_format;

	friend std::ostream& operator<<(std::ostream& os,
			const FaultDescriptor& fd) {
		os << "Min relative: " << fd.min_relative_error << std::endl;
		os << "Max relative: " << fd.max_relative_error << std::endl;
		os << "Layer i: " << fd.layer_i << " Layer format: "
				<< fd.geometry_format;
		return os;
	}

	friend std::istream& operator>>(std::istream& is, FaultDescriptor& fd) {
		//Read the parameters from file
		is >> fd.min_relative_error;
		is >> fd.max_relative_error;
		is >> fd.geometry_format;
		is >> fd.layer_i;
		return is;
	}
};

template<typename str_t>
std::string get_enviroment_var(str_t& src) {
	auto ptr = std::getenv(src);
	std::string ret = "";
	if (ptr) {
		ret = std::string(ptr);
	}
	return ret;
}

void simulate_scheduler_fault(int M, int N, int layer_count_output,
		std::vector<float>& C) {
	std::string fault_parameter_file_path = get_enviroment_var(
			"FAULT_PARAMETER_FILE");

	if (!fault_parameter_file_path.empty()) {
		//Open the fault injection archive
		std::ifstream parameter_file(fault_parameter_file_path);
		/**
		 * For random selection
		 */
		//Will be used to obtain a seed for the random number engine
		std::random_device rd;
		//Standard mersenne_twister_engine seeded with rd()
		std::mt19937 gen(rd());

		if (parameter_file.good()) {
			FaultDescriptor fd;
			parameter_file >> fd;
			parameter_file.close();

			std::uniform_real_distribution<float> float_generator(
					fd.min_relative_error, fd.max_relative_error);
			std::uniform_int_distribution<int> bool_generator(0, 1);

			if (fd.layer_i == layer_count_output) {
				if (DEBUG >= 1) {
					std::cout << "DEBUG MIN RELATIVE" << std::endl;
					std::cout << fd << std::endl;
				}

				//size selection
				std::uniform_int_distribution<int> int_m_generator(0,
						M - (BLOCK_SIZE < M ? BLOCK_SIZE : 0));
				std::uniform_int_distribution<int> int_p_generator(0,
						N - (BLOCK_SIZE < N ? BLOCK_SIZE : 0));
				auto start_i = int_m_generator(gen);
				auto start_j = int_p_generator(gen);

				if (fd.geometry_format == "RANDOM") {

					for (auto i = start_i; i < M; i++) {
						for (auto j = start_j; j < N; j++) {
							auto is_necessary_to_inject = bool(
									bool_generator(gen));
							if (is_necessary_to_inject) {
								C[i * N + j] *= float_generator(gen);
							}
						}
					}
				} else if (fd.geometry_format == "SQUARE") {
					for (auto i = start_i; i < M; i++) {
						for (auto j = start_j; j < N; j++) {
							C[i * N + j] *= float_generator(gen);
						}
					}
				} else if (fd.geometry_format == "ALL") {
					for (auto i = 0; i < M; i++) {
						for (auto j = 0; j < N; j++) {
							C[i * N + j] *= float_generator(gen);
						}
					}
				} else if (fd.geometry_format == "LINE") {
					auto col_or_line = bool(bool_generator(gen));

					if (col_or_line) {
						//select a line
						std::uniform_int_distribution<int> int_m_generator_line(0,
								M - 1);
						auto i = int_m_generator_line(gen);

						for (auto j = 0; j < N; j++) {
							C[i * N + j] *= float_generator(gen);
						}
					} else {
						//select a line
						std::uniform_int_distribution<int> int_n_generator_column(0,
								N - 1);
						auto j = int_n_generator_column(gen);
						for (auto i = 0; i < M; i++) {
							C[i * N + j] *= float_generator(gen);
						}
					}
				}
			}
		} else {
			throw std::runtime_error(
					"COULD NOT OPEN FILE: " + fault_parameter_file_path);
		}
	}
}

void parse_output_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float *C_gpu) {
	if (FLEX_GRIP_ANALYSIS != 1) {
		return;
	}

	static int layer_count_output = 0;
	if (reset_counters_var) {
		layer_count_output = 0;
		reset_counters_var = false;
	}
	layer_count_output++;
	if (DEBUG >= 1)
		std::cout << "Layer " << layer_count_output << std::endl;
	/**
	 * If A is an m × n matrix and B is an n × p matrix,
	 * C is the m × p matrix
	 */
	auto size_c = M * N;

	std::vector<float> C_cpu(size_c);
	cuda_pull_array(C_gpu, C_cpu.data(), size_c);

	switch (operation_type) {
	case SIMULATE_SCHEDULER_FAULT: {
		if (DEBUG >= 2)
			std::cout << "Scheduler faults selected\n";
		simulate_scheduler_fault(M, N, layer_count_output, C_cpu);
		break;
	}

	case INVALID_OPTION:
		if (DEBUG == 1)
			std::cout << "Invalid option\n";
		break;
	}

	cuda_push_array(C_gpu, C_cpu.data(), size_c);
}

void parse_input_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C,
		int ldc) {
	if (FLEX_GRIP_ANALYSIS != 1) {
		return;
	}
}

