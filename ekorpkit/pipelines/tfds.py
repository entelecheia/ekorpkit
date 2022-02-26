import tensorflow as tf
import tensorflow_io as tfio


def get_dataset(path_to_files):
    autotune = tf.data.experimental.AUTOTUNE
    filenames = tf.data.Dataset.list_files(path_to_files + "/*", shuffle=True)

    def parquet_ds(file):
        ds = tfio.IODataset.from_parquet(file, {"image": tf.string, "label": tf.int32})
        return ds

    ds = filenames.interleave(
        parquet_ds, num_parallel_calls=autotune, deterministic=False
    )

    def parse(example):
        image = tf.io.decode_raw(example["image"], tf.uint8)
        image = tf.reshape(image, [32, 32, 3])
        label = example["label"]
        return image, label

    ds = ds.map(parse, num_parallel_calls=autotune)

    return ds
