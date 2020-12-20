# Installation

## Visual Studio

Install TF.NET:

```bash
### install tensorflow C#/F# binding
PM> Install-Package TensorFlow.NET
### install keras for tensorflow
PM> Install-Package TensorFlow.Keras
```

Install TensorFlow binary (**essential**). Choose one of the following:

```bash
### Install tensorflow binary
### For CPU version
PM> Install-Package SciSharp.TensorFlow.Redist -Version 2.3.1

### For GPU version (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU -Version 2.3.1
```

## dotnet CLI

Install TF.NET:

```bash
### install tensorflow C#/F# binding
dotnet add package TensorFlow.NET
### install keras for tensorflow
dotnet add package TensorFlow.Keras
```

Install TensorFlow binary (**essential**). Choose one of the following:

```bash
### Install tensorflow binary
### For CPU version
dotnet add package SciSharp.TensorFlow.Redist -v 2.3.1

### For GPU version (CUDA and cuDNN are required)
dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU -v 2.3.1

```

> ...noted that the version digits of the binary (SciSharp.TensorFlow.Redist\*) is the same as that of Google's TensorFlow.
>
> ...if GPU package is chosen, **make very sure** that if the versions of TensorFlow (i.e. `tf native`) and CUDA are compatible:
>
> | TF.NET \ TensorFlow       | tf native 1.14, cuda 10.0 | tf native 1.15, cuda 10.0 | tf native 2.3, cuda 10.1 | tf native 2.4, cuda 11 |
> | ------------------------- | :----------------------: | :-----------------------: | :----------------------: | :--------------------: |
> | tf.net 0.3x, tf.keras 0.2 |                          |                           |            x             |     not compatible     |
> | tf.net 0.2x               |                          |             x             |            x             |                        |
> | tf.net 0.15               |            x             |             x             |                          |                        |
> | tf.net 0.14               |            x             |                           |                          |                        |

## Troubleshooting

Still got some problems?

-   [More detailed local documentation](essentials/installationTroubleshooting.md).
-   [Microsoft official documentation](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.vision.imageclassificationtrainer?view=ml-dotnet#using-tensorflow-based-apis).
-   [Step-by-step setup tutorial](https://medium.com/dev-genius/tensorflow-basic-setup-for-net-developers-d56bfb0af40e).
-   Or [post](https://github.com/SciSharp/TensorFlow.NET/issues) an issue for help!
