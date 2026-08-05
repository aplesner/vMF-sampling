// CPU-only PyTorch binding for CUSF's log modified-Bessel implementation.
// This intentionally builds without CUDA so it can be used on macOS and on
// CPU-only benchmark nodes.

#include <torch/extension.h>

#define __CUSF_DEVICE__ cpu
#include <bessel/iv_log.h>

namespace {

torch::Tensor iv_log(torch::Tensor order, torch::Tensor x) {
    TORCH_CHECK(order.device().is_cpu(), "order must be a CPU tensor");
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(order.scalar_type() == x.scalar_type(), "order and x must have the same dtype");
    TORCH_CHECK(order.sizes() == x.sizes(), "order and x must have the same shape");
    TORCH_CHECK(order.is_contiguous() && x.is_contiguous(), "order and x must be contiguous");

    auto output = torch::empty_like(x);
    switch (x.scalar_type()) {
        case torch::ScalarType::Float:
            cusf::cpu::bessel::iv_log<float>(
                order.data_ptr<float>(), x.data_ptr<float>(),
                output.data_ptr<float>(), x.numel());
            break;
        case torch::ScalarType::Double:
            cusf::cpu::bessel::iv_log<double>(
                order.data_ptr<double>(), x.data_ptr<double>(),
                output.data_ptr<double>(), x.numel());
            break;
        default:
            TORCH_CHECK(false, "CUSF CPU iv_log supports float32 and float64");
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("iv_log", &iv_log, "CUSF CPU log(I_v(x))");
}
