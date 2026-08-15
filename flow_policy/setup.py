from setuptools import find_packages, setup

setup(
    name='diffusion_policy_3d',
    packages=find_packages(),
    install_requires=[
        'zarr==2.16.1',
        'h5py==3.11.0',
        'scipy==1.10.1'
    ]
)
