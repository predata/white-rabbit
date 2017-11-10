from setuptools import setup, find_packages

setup(name="white_rabbit",
      version="0.0.8",
      description="Vectorized summary stats computation",
      url="https://github.com/predata/white-rabbit",
      author="Predata",
      author_email="dev@predata.com",
      packages=find_packages(),
      install_requires=[
          'numpy>=1.11.0',
          'pandas>=0.18.0'
      ],
     )
