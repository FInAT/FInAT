from distutils.core import setup
import sys

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)


setup(name="FInAT",
      version="0.1",
      description="FInAT Is not A Tabulator",
      author="Imperial College London and others",
      author_email="david.ham@imperial.ac.uk",
      url="https://github.com/FInAT/FInAT",
      license="MIT",
      packages=["finat"])
