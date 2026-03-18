from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="kvcache_compression",
    version="0.1.0",
    packages=find_packages(include=["int4_ext", "int4_ext.*", "quant", "quant.*", "huffman", "huffman.*"]),
    ext_modules=[
        CUDAExtension(
            name="int4_ext._C",
            sources=["int4_ext/int4_ext.cpp", "int4_ext/int4_ext_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
