#!/bin/bash -e

# Set up an environment to run CI tests, e.g. with GitHub Actions or Travis

if [ $# -ne 2 ]; then
  echo "Usage: $0 top_directory python_version"
  exit 1
fi

top_dir=$1
python_version=$2
bin_dir=${top_dir}/bin
mkdir -p ${bin_dir}

conda config --remove channels defaults  # get conda-forge, not main, packages
conda create --yes -q -n python${python_version} -c salilab -c conda-forge python=${python_version} pip biopython networkx dill pandas scipy matplotlib imp-nightly eigen swig cmake
eval "$(conda shell.bash hook)"
conda activate python${python_version}
pip install pytest-cov coverage flake8

temp_dir=`mktemp -d`
cd ${temp_dir}

# STRIDE
if [ ! -e ${bin_dir}/stride ]; then
  wget http://webclu.bio.wzw.tum.de/stride/stride.tar.gz
  tar -xvzf stride.tar.gz
  make
  mv stride ${bin_dir}/stride
fi

cd /
rm -rf ${temp_dir}
