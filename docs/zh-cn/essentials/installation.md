# 安装

## Visual Studio

安装 TF.NET:

```bash
### install tensorflow C#/F# binding
PM> Install-Package TensorFlow.NET
### install keras for tensorflow
PM> Install-Package TensorFlow.Keras
```

安装 TensorFlow binary (**必要**). 在下面选择其一运行:

```bash
### Install tensorflow binary
### For CPU version
PM> Install-Package SciSharp.TensorFlow.Redist

### For GPU version (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
```

## dotnet CLI

安装 TF.NET:

```bash
### install tensorflow C#/F# binding
dotnet add package TensorFlow.NET
### install keras for tensorflow
dotnet add package TensorFlow.Keras
```

安装 TensorFlow binary (**必要**). 在下面选择其一运行:

```bash
### Install tensorflow binary
### For CPU version
dotnet add package SciSharp.TensorFlow.Redist -v 2.3.1

### For GPU version (CUDA and cuDNN are required)
dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU -v 2.3.1
```

> ……binary (SciSharp.TensorFlow.Redist\*) 的版本号跟 Google 的 TensorFlow 是一致的。
>
> ……如果选了 GPU 依赖, **必须**确认 TensorFlow (i.e. `tf native`) 和 CUDA 的版本是能够兼容的:
>
> | TF.NET \ TensorFlow       | tf native 1.14, cuda 10.0 | tf native 1.15, cuda 10.0 | tf native 2.3, cuda 10.1 | tf native 2.4, cuda 11 |
> | ------------------------- | :----------------------: | :-----------------------: | :----------------------: | :--------------------: |
> | tf.net 0.3x, tf.keras 0.2 |                          |                           |            x             |     not compatible     |
> | tf.net 0.2x               |                          |             x             |            x             |                        |
> | tf.net 0.15               |            x             |             x             |                          |                        |
> | tf.net 0.14               |            x             |                           |                          |                        |

## 疑难杂症

仍被问题所困扰？

-   到 [更深入的技术文档](essentials/installationTroubleshooting.md) 看看有没有解决方法。
-   [微软官方文档](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.ml.vision.imageclassificationtrainer?view=ml-dotnet#using-tensorflow-based-apis).
-   [手把手安装教程（英文）](https://medium.com/dev-genius/tensorflow-basic-setup-for-net-developers-d56bfb0af40e).
-   或者干脆 [发个 issue](https://github.com/SciSharp/TensorFlow.NET/issues) 找人帮忙看看。
