#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
	Parameters:
	- Input: A,
	- weights: W,
	- bias: b
	- Output: Z,

*/

namespace {
	template <typename scalar_t>
	__device__ __forceinline__ scalar_t matmul(scalar_t A[N][D], // Input
											   scalar_t W[D][M], // Weights
											   scalar_t b[N][M], // Bias
											   scalar_t alpha, // A*W
											   scalar_t beta) { // bias
		auto Cvalue = 0;
		auto C[][] = 0;
		auto row = blockIdx.y * blockDim.y + threadIdx.y; // j
		auto col = blockIdx.x * blockDim.x + threadIdx.x; // i
		for (auto e = 0; e < A.width; ++e)
			Cvalue += A.elements[row * A.width + e]
					* W.elements[e * W.width + col];
		C.elements[row * C.width + col] = Cvalue; // Input * weights
		if (col < N && row < N)
			return (alpha * C[col][row] + beta * b[col][row]); // D: bias
	}

	template <typename scalar_t>
	__device__ __forceinline__ scalar_t transpose(scalar_t A[N][M]) {
		auto Cvalue = 0;
		auto C[][] = 0;
		auto row = blockIdx.y * blockDim.y + threadIdx.y; // y
		auto col = blockIdx.x * blockDim.x + threadIdx.x; // x
		
		if (col < N && row < N) {
			auto index_in = col + N * row;
			auto index_out = row + N * col;
			return C[index_out] = A[index_in];
		}
	}

	template <typename scalar_t>
	__device__ __forceinline__ rowSum(scalar_t A[R][C]) {

		int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
		if (rowIdx < R) {
			float sum=0;
			for (int k = 0 ; k < C ; k++)
				sum += m[rowIdx * C + k];
			s[rowIdx] = sum;            
		}
	
	}

	template <typename scalar_t>
	__global__ void FullyConnectedLayerForward_kernel(
		torch::Tensor A,
		torch::Tensor W,
		torch::Tensor b
		torch::Tensor Z) {
			Z = matmul(A, W, b, 1, 1);
	}

	__global__ void FullyConnectedLayerBackward_kernel(
		torch::Tensor A,
		torch::Tensor W,
		torch::Tensor b,
		torch::Tensor dZ,
		torch::Tensor dA,
		torch::Tensor dW,
		torch::Tensor db) {
			dA = matmul(transpose(W), dZ, A, 1, 0);
			dW = matmul(dZ, A, W, 1, 0);
			db = rowSum(dZ);
	}
}


std::vector<torch::Tensor> fc_layer_cuda_forward(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor b,
    torch::Tensor Z) {
		const int threads = 1024;
		const auto batch_size = A.size(0);
		const auto state_size = A.size(1);
		const dim3 blocks((state_size + threads - 1) / threads, batch_size);
		
		FullyConnectedLayerForward_kernel<<<blocks, threads>>>(A, W, b, Z)
		return {Z, A, W, b};
	}

std::vector<torch::Tensor> fc_layer_cuda_backward(
	torch::Tensor A,
	torch::Tensor W,
	torch::Tensor b,
	torch::Tensor dZ,
	torch::Tensor dA,
	torch::Tensor dW,
	torch::Tensor db) {
		const int threads = 1024;
		const auto batch_size = A.size(0);
		const auto state_size = A.size(1);
		const dim3 blocks((state_size + threads - 1) / threads, batch_size);
		
		FullyConnectedLayerBackward_kernel<<<blocks, threads>>>(A, W, b, dZ, dA, dW, db)
		return {dA, dW, db};
	}
