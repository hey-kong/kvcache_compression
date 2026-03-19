from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="kvcache_compression",
    version="0.1.0",
    packages=find_packages(include=["quant", "quant.*", "huffman", "huffman.*"]),
    ext_modules=[
        CppExtension(
            name="huffman._C",
            sources=["huffman/huffman_ext.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
            },
        ),
        CUDAExtension(
            name="quant.int4_ext._C",
            sources=["quant/int4_ext/int4_ext.cpp", "quant/int4_ext/int4_ext_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
