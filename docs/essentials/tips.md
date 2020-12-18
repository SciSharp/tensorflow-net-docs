# Tips

## Use functions from `tf` if possible

Math operations on `np` is **not stable** for now. Try those alternatives in `tf` if possible.

For example, use this:

```csharp
tf.matmul();
```

...rather than this:

```csharp
np.matmul();
```

## Use `np` if saving/loading `Tensor`

## Use the latest TF.NET version

Since TF.NET version correlates that of native TensorFlow, which in turn links to the version of CUDA, it is inconvenient. Though, it is a mission impossible for the small team to maintain all the TF.NET versions. Please use the latest package if possible.
