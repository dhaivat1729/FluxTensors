# Import the setup function from setuptools, which is a Python package used for distributing Python packages.
from setuptools import setup

# Import BuildExtension and CUDAExtension from torch.utils.cpp_extension.
# BuildExtension is a custom command for building C++ extensions in PyTorch.
# CUDAExtension is a helper class for building CUDA extensions.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Call the setup function to define the package details and the extension modules.
setup(
    # The name of the package being created.
    name='FluxTensors',
    version='0.1',
    description='Linear Layer in CUDA',
    author='Dhaivat Bhatt',
    author_email='dhaivat1994@gmail.com',
    url='https://github.com/dhaivat1994/FluxTensors',
    packages=['linear_layer'],
    package_dir={'linear_layer': 'src'},
    # Define the extension modules to be built.
    ext_modules=[
        CUDAExtension(
            name='linear_layer',
            sources=['/home/ubuntu/LLM_env/FluxTensors/src/linear_layer.cu'],
            include_dirs=['/home/ubuntu/LLM_env/FluxTensors/include'],  # Ensure this is correct
            extra_compile_args={'cxx': ['-g', '-std=c++14'],
                                'nvcc': ['-O2']}
        ),
    ],
    
    # Specify the command class to use for building the extension.
    # Here, we are using BuildExtension to handle the build process.
    cmdclass={'build_ext': BuildExtension}
)
    