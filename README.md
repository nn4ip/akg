> Dev branch off commit d6d6edc (v1.6.0).

- [What Is AKG?](#what-is-akg)
- [Hardware Backends Support](#hardware-backends-support)
- [Build](#build)
    - [Build With MindSpore](#build-with-mindspore)
    - [Build Standalone](#build-standalone)
- [Run](#run)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

[查看中文](./README_CN.md)

## What Is AKG
AKG(Auto Kernel Generator) is an optimizer for operators in Deep Learning Networks. It provides the ability to automatically fuse ops with specific patterns. AKG works with MindSpore-GraphKernel to improve the performance of networks running on different hardware backends.

AKG composes with three basic optimization module, normalization, auto schedule and backend optimization.
- **normalization.** In order to solve the limitation in expression ability of polyhedral(which can only process static linear programs), the computation IR needs to be normalized first. The mainly optimization of normalization module includes auto-inline, loop fusing, common subexpression elimination and so on.
- **auto schedule.** Base on polyhedral technology, the auto schedule module mainly have auto-vectorization, auto-tiling, thread/block mapping, dependency analysis and memory promotion.
- **backend optimization.** The backend optimization module mainly consists of TensorCore acceleration, double buffer optimization, storage flatten optimization and inject sync optimization.

  <img src="docs/akg-design.png" style="zoom:80%" div align=center/>

## Hardware Backends Support
At present, `Ascend910`, `NVIDIA V100/A100` and `CPU` are supported. More Backends are on the list.

## Build

### Build With MindSpore
See [MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md) for details.

### Build Standalone
We suggest you build and run akg together with MindSpore. And we also provide a way to run case in standalone mode for convenience sake.
Refer to [MindSpore Installation](https://www.mindspore.cn/install/en) for more information about compilation dependencies.
- Build on Ascend910

  [git-lfs](https://github.com/git-lfs/git-lfs/wiki/installation) needs to be installed before cloning the source codes.
  ```
  git clone https://gitee.com/mindspore/akg.git
  cd akg
  bash build.sh -e ascend -j8
  ```

- Build on GPU
  ```
  git clone https://gitee.com/mindspore/akg.git
  cd akg
  bash build.sh -e gpu -j8
  ```

- Build on CPU
  ```
  git clone https://gitee.com/mindspore/akg.git
  cd akg
  bash build.sh -e cpu -j8
  ```

## Run Standalone
1. Set Environment

- Ascend910
  ```
  cd tests
  source ./test_env.sh
  ```

- NVIDIA V100/A100
  ```
  cd tests
  source ./test_env.sh gpu
  ```

- CPU V100/A100
  ```
  cd tests
  source ./test_env.sh cpu
  ```

2. Run test

- Use script:
```
cd tests/st
python run.py -e gpu -o add -l level0  # run add operator on GPU
```
  Detailed instructions see:`python run.py -h`
- Use specific case:

  - Ascend910
  ```
  cd tests/st/ops/
  pytest -s test_abs.py -m "level0 and platform_x86_ascend_training" # run level0 testcases on Ascend
  ```

  - NVIDIA V100/A100
  ```
  cd tests/st/ops/
  pytest -s test_abs.py -m "level0 and platform_x86_gpu_training" # run level0 testcases on GPU
  ```

  - CPU
  ```
  cd tests/st/ops/
  pytest -s test_abs.py -m "level0 and platform_x86_cpu" # run level0 testcases on CPU
  ```

## Using AKG to generate high performance kernels
See [Wiki](https://gitee.com/mindspore/akg/wikis).

## Contributing

Welcome contributions. See [MindSpore Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
