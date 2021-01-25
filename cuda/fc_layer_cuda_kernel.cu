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

__global__ void FullyConnectedLayerForward_kernel(
	torch::Tensor A,
	torch::Tensor W,
	torch::Tensor b,
	torch::Tensor Z) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A.size(1);
	int Z_y_dim = W.size(2);

	auto Z_value = torch::zeros_like(Z);

	if (row < Z_y_dim && col < Z_x_dim) {
		for (int i = 0; i < W.size(1); i++) {
			Z_value += W[row * W.size(1) + i] * A[i * Z_x_dim + col];
		}
		Z[row * Z_x_dim + col] = Z_value + b[row];
	}
}

__global__ void FullyConnectedLayerBackward_kernel(
	torch::Tensor A,
	torch::Tensor W,
	torch::Tensor b,
	torch::Tensor dZ,
	torch::Tensor dA,
	torch::Tensor dW,
	torch::Tensor db) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// W is treated as transposed
	int dA_x_dim = dZ.size(1);
	int dA_y_dim = dW.size(1);

	auto dA_value = torch::zeros_like(dA);
	auto dW_value = torch::zeros_like(dW);

	if (row < dA_y_dim && col < dA_x_dim) {
		for (int i = 0; i < W.size(2); i++) {
			dA_value += W[i * W.size(1) + row] * dZ[i * dZ.size(1) + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
	if (row < W.size(2) && col < W.size(1)) {
		for (int i = 0; i < dZ.size(1); i++) {
			dW_value += dZ[row * dZ.size(1) + i] * A[col * dA_x_dim + i];
		}
		dW[row * W.size(1) + col] = dW_value;
	}
	if (col < dZ.size(1) * dZ.size(2)) {
		int dZ_x = col % dZ.size(1);
		int dZ_y = col / dZ.size(1);
		db[col * dZ.size(2) + row] += dZ[dZ_y * dZ.size(1) + dZ_x]
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
		
		FullyConnectedLayerForward_kernel<<<blocks, threads>>>(
			A,
			W,
			b,
			Z
		)
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
		
		FullyConnectedLayerBackward_kernel<<<blocks, threads>>>(
			A,
			W,
			b,
			dZ,
			dA,
			dW,
			db
		)
		return {dA, dW, db};
	}
