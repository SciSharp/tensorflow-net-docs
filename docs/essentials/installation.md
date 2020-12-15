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
PM> Install-Package SciSharp.TensorFlow.Redist

### For GPU version (CUDA and cuDNN are required)
PM> Install-Package SciSharp.TensorFlow.Redist-Windows-GPU
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
dotnet add package SciSharp.TensorFlow.Redist

### For GPU version (CUDA and cuDNN are required)
dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU
```
