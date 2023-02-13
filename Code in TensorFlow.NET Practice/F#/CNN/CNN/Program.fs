open System.IO
open Tensorflow
open Tensorflow.Keras
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Utils
open type Tensorflow.Binding
open type Tensorflow.KerasApi

let batch_size = 32
let epochs = 3
let img_dim = Shape (180, 180)

let private prepareData () =
    let fileName = "flower_photos.tgz"
    let url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    let data_dir = Path.Combine(Path.GetTempPath(), "flower_photos")
    Web.Download(url, data_dir, fileName) |> ignore
    Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir)
    let data_dir = Path.Combine(data_dir, "flower_photos")

    // convert to tensor
    let train_ds =
        keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2f,
            subset = "training",
            seed = 123,
            image_size = img_dim,
            batch_size = batch_size)

    let val_ds =
        keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2f,
            subset = "validation",
            seed = 123,
            image_size = img_dim,
            batch_size = batch_size)

    let train_ds = train_ds.shuffle(1000).prefetch(buffer_size = -1)
    let val_ds = val_ds.prefetch(buffer_size = -1)

    for img, label in train_ds do
        print($"images: {img.TensorShape}")
        print($"labels: {label.numpy()}")

    train_ds, val_ds

let private buildModel () =
    let num_classes = 5
    let layers = keras.layers
    let model = keras.Sequential(ResizeArray<ILayer>(seq {
        layers.Rescaling(1.0f / 255f, input_shape = Shape ((int)img_dim.dims.[0], (int)img_dim.dims.[1], 3)) :> ILayer;
        layers.Conv2D(16, Shape 3, padding = "same", activation = keras.activations.Relu);
        layers.MaxPooling2D();
        //layers.Conv2D(32, Shape 3, padding = "same", activation = keras.activations.Relu);
        //layers.MaxPooling2D();
        //layers.Conv2D(64, Shape 3, padding = "same", activation = keras.activations.Relu);
        //layers.MaxPooling2D();
        layers.Flatten();
        layers.Dense(128, activation = keras.activations.Relu);
        layers.Dense(num_classes) }))

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = true),
        metrics = [| "accuracy" |])

    model.summary()
    model

let private train train_ds val_ds (model : Sequential) =
    model.fit(train_ds, validation_data = val_ds, epochs = epochs)

let private run () =
    let train_ds, val_ds = prepareData()
    let model = buildModel()
    train train_ds val_ds model

run()