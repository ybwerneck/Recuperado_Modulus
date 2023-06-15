# ⛔️ Modulus is moving to GitHub! Please check out our new [GitHub Repos](https://github.com/nvidia/modulus). This repo will no longer be supported and will be deprecated by April 30, 2023.  

# Modulus Documentation

Modulus documentation includes user guide, installation instructions, release notes and API documentation.
Built using Sphinx, contributing to the Modulus Documentation is a straight forward process given that all defined standards are followed.

## Builds

#### [Nightly](https://modulus.gitlab-master-pages.nvidia.com/docs/) | [Release Stage](https://sw-docs-dgx-station.nvidia.com/deeplearning/modulus/) | [Public](https://docs.nvidia.com/deeplearning/modulus/)

## Building from Source

### Cloning this Repository

Modulus documentation relies on two git submodules (external/Modulus and external/examples) for building API documentation and code block, respectively.
To properly clone this repo and initialize the submodules correctly, we suggest the following process:

1. Clone the docs repo. The master branch currently does not have any submodules. The below command will clone the `develop` branch
```
git clone https://gitlab-master.nvidia.com/modulus/docs.git
cd docs/
```
2. The example folder will be created but will be empty. To populate it correctly, please do the below:
```
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init
```
*Note*: `GIT_LFS_SKIP_SMUDGE=1` will skip any LFS files, which should not be included in documentation.

To reset your submodules after pulling use:
```
GIT_LFS_SKIP_SMUDGE=1 git submodule foreach git reset --hard
```

3. This will pull the latest commit for the examples/modulus repo that was tagged in the docs project. If you have made any changes after the docs project was updated, you can go to the examples folder, and do a `git pull` to update it. As an example:
```
cd external/examples/
git pull origin develop
```

### Running Sphinx

Building the documentation requires the use of the docker image supplied.
Until further automated, the following build process is recommended.

1. Start by pulling the latest documentation docker image if not already installed:
```
docker login gitlab-master.nvidia.com:5005
docker pull gitlab-master.nvidia.com:5005/modulus/docs:22.09
```

2. Launch a docker container from the current docker image and mount the documentation repository:
```
docker run -i -v ${PWD}:/docs/ -t gitlab-master.nvidia.com:5005/modulus/docs:22.09
```

3. Install current Modulus package locally in docker container for building the API docs:
```
make install
```

4. Build the HTML documentation with Sphinx. This command should be repeated to have changes in the .rst files be reflected in your local HTML files:
```
make html
```

This will place the HTML files in the `_build` folder. Sometimes edits require the docs to be completely rebuilt for which `make clean` can be used to remove old build files.


## Resources:

Editing the .rst files can be found at below resources:

https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#restructured-text-rest-and-sphinx-cheatsheet

