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
