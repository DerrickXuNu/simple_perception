from setuptools import setup, find_packages

setup(
  name ='simple-perception',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Make complicated AV perception algorithms easy to learn, explore and use',
  long_description_content_type = 'text/markdown',
  author = 'Runsheng Xu',
  author_email = 'rxx3386@ucla.edu',
  url = 'https://github.com/DerrickXuNu/simple_perception',
  keywords = [
    'artificial intelligence',
    'autonomous driving',
    'object detection'
  ],
  install_requires=[
    'einops>=0.6.0',
    'torch==1.12.0',
     'scipy',
    'torchvision==0.13.0'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==1.12.0',
    'torchvision==0.13.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)