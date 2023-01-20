from setuptools import setup, find_packages

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = '0.31.1',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'accelerate',
    'dataclasses',
    'einops',
    'ema-pytorch',
    'importlib_metadata',
    'matplotlib',
    'packaging',
    'pandas',
    'pillow',
    'ruamel.yaml',
    'torch',
    'torchmetrics==0.8.2',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
