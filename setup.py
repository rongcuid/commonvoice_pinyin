from setuptools import setup, find_packages

setup(name="commonvoice_pinyin",
      version="1.0",
      url="",
      author="Rongcui Dong",
      author_email="rongcuid@outlook.com",
      license="MIT",
      packages=find_packages(),
      install_requires=[
        "numpy<1.21", # numba
        "torch",
        "torchaudio",
        "pandas",
        "diskcache",
        "librosa",
        "pypinyin",
      ],
      zip_safe=False
      )
