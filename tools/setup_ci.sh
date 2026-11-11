#!/bin/bash -e

# Set up an environment to run CI tests, e.g. with GitHub Actions or Travis

if [ $# -ne 1 ]; then
  echo "Usage: $0 top_directory"
  exit 1
fi

top_dir=$1
bin_dir=${top_dir}/bin
mkdir -p ${bin_dir}

eval "$(conda shell.bash hook)"
conda activate test
conda install --yes -q pip biopython networkx dill pandas scipy matplotlib imp-nightly eigen swig cmake
pip install pytest-cov coverage flake8

temp_dir=`mktemp -d`
cd ${temp_dir}

# STRIDE
if [ ! -e ${bin_dir}/stride ]; then
  wget https://salilab.org/stride/Stride_src.tar.Z
  tar -xvzf Stride_src.tar.Z
  make
  mv stride ${bin_dir}/stride
fi

cd /
rm -rf ${temp_dir}
