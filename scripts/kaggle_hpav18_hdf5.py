import os
import h5py as h5
import zipfile
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def arguments():
    parser = ArgumentParser(
        description=(
            "Merges the HPA Kaggle dataset and HPAv18 external data into a "
            "single HDF5 file"
        )
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        type=str,
        default="../../dataset",
        help="Path to the root directory of the Kaggle dataset",
    )
    parser.add_argument(
        "--dest-dir",
        "-d",
        type=str,
        default="../../dataset",
        help="Path to the root directory where the HDF5 files will be saved",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=20,
        help="Number of images read at a time",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=1,
        help="Number of images in each chunk of the HDF5 file",
    )
    parser.add_argument(
        "--no-compression",
        dest="compression",
        action="store_false",
        help=(
            "Doesn't apply any compression filter to the HDF5 files. If not specified, "
            "LZF compression is applied"
        ),
    )
    parser.add_argument(
        "--compression-filter",
        "-f",
        choices=["gzip", "szip", "lzf"],
        default="lzf",
        help="Legal values are gzip, szip, and lzf. Default: lzf",
    )
    parser.add_argument(
        "--extra-data",
        "-x",
        action="store_true",
        help="Includes extra data in the training HDF5 file",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Testing purposes. Total number of images to store in the HDF5 files.",
    )

    return parser.parse_args()


def get_image(zip_file, name, filters=("red", "green", "blue", "yellow")):
    """Gets a numpy array with the specified filters given the file name.

    Arguments:
        zip_file (zipfile.ZipFile): a zip archive containing the images.
        name (str): the name of the image to retrieve from ``zip_file``. The names must
            not contain the filter.
        filters (array-like of str, optional): the filters to concatenate for each
            image. Default: ("red", "green", "blue", "yellow").

    Returns:
        numpy.ndarray: the image with shape (H, W, F) where H, W, and F are image
            height, width, and number fo filters, respectively.

    """
    # Iterate over the list of filters and load them one-by-one
    img_filters = []
    for f in filters:
        img_name = name + "_" + f + ".png"
        zip_img = zip_file.open(img_name)
        img = Image.open(zip_img)
        img_np = np.asarray(img, dtype=np.uint8)
        img_filters.append(img_np)

    img_np = np.stack(img_filters, axis=-1).squeeze()

    return img_np


def image_batches(
    zip_file, image_names, batch_size=200, filters=("red", "green", "blue", "yellow")
):
    """Generates batches of images given their directory and name (without filter)

    Arguments:
        zip_file (zipfile.ZipFile): a zip archive containing the images.
        image_names (array-like of str): a list of names of images to retrieve from
            ``zip_file``. The names must not contain the filter.
        batch_size (int, optional): the number of images to return per batch.
            Default: 200.
        filters (array-like of str, optional): the filters to concatenate for each
            image. Default: ("red", "green", "blue", "yellow").

    Returns:
        numpy.ndarray: a batch of images with shape (batch_size, H, W, F) where H, W,
            and F are image height, width, and number fo filters, respectively.

    """
    batch = []
    batch_names = []
    for idx, name in enumerate(image_names):
        batch.append(get_image(zip_file, name))
        batch_names.append(name)

        # Return the accumulated images and names when batch_size is reached or if
        # this is the last iteration
        if len(batch) == batch_size or idx == len(image_names) - 1:
            # h5py doesn't support Unicode strings so the names have to be encoded to
            # UTF-8 which changes the type to zero-terminated byte typestrings
            names = np.chararray.encode(np.array(batch_names), encoding="utf8")
            images = np.array(batch)

            # Reset lists
            batch = []
            batch_names = []

            yield names, images


def hdf5_writer(
    zip_path,
    hdf5_path,
    batch_size,
    chunk_size,
    compression=None,
    max_limit=None,
    append=False,
):
    # Open the zip file and get the list of images to store in the HDF5 file
    archive = zipfile.ZipFile(zip_path, "r")
    zip_names = archive.namelist()

    # image_batches expects the image names to not include the filters. After removing
    # the filters also remove the duplicated names and sort the list
    zip_names = [name.rsplit("_", 1)[0] for name in zip_names]
    zip_names = list(sorted(set(zip_names)))
    zip_names = zip_names[:max_limit]

    with tqdm(zip_names) as pbar:
        # Iterate over the batches and add them to the hdf5 file
        batch_loader = image_batches(archive, zip_names, batch_size=batch_size)
        for idx, (names, images) in enumerate(batch_loader):
            if idx == 0 and not append:
                # On the first iteration create the file and two datasets. One
                # contains the images and the other their respective names
                with h5.File(hdf5_path, "w") as f:
                    f.create_dataset(
                        "images",
                        data=images,
                        maxshape=(None, *images.shape[1:]),
                        chunks=(chunk_size, *images.shape[1:]),
                        compression=compression,
                    )
                    f.create_dataset(
                        "names",
                        data=names,
                        maxshape=(None,),
                        chunks=(chunk_size,),
                        compression=compression,
                    )
            else:
                # Append the new images and names to the existing hdf5 file
                with h5.File(hdf5_path, "a") as f:
                    f["images"].resize(f["images"].shape[0] + len(images), axis=0)
                    f["images"][-images.shape[0] :] = images

                    f["names"].resize(f["names"].shape[0] + len(names), axis=0)
                    f["names"][-len(names) :] = names

            pbar.update(len(images))


if __name__ == "__main__":
    # Parameters
    args = arguments()
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    if args.compression:
        compression_filter = args.compression_filter
    else:
        compression_filter = None

    filters = ("red", "green", "blue", "yellow")

    # Just for testing; set to None to store all images in hdf5
    if args.num_images == 0:
        num_images = None
    else:
        num_images = args.num_images

    # Relevant paths for the kaggle dataset
    train_zip = os.path.join(source_dir, "train.zip")
    test_zip = os.path.join(source_dir, "test.zip")
    csv_path = os.path.join(source_dir, "train.csv")

    # Output files
    train_hdf5 = os.path.join(dest_dir, "train.hdf5")
    test_hdf5 = os.path.join(dest_dir, "test.hdf5")

    print("Creating an HDF5 file for the training set")
    print("Loading from:", train_zip)
    print("HDF5 file will be stored at:", train_hdf5)
    hdf5_writer(
        train_zip,
        train_hdf5,
        batch_size,
        chunk_size,
        compression_filter,
        max_limit=num_images,
    )

    if args.extra_data:
        # Relevant paths for the external data
        hpav18_train_zip = os.path.join(source_dir, "HPAv18_train.zip")
        hpav18_csv_path = os.path.join(source_dir, "HPAv18_train.csv")
        merged_csv = os.path.join(dest_dir, "kaggle_HPAv18_train.csv")

        # Add external data to the existing HDF5 file
        print()
        print("Adding extra data to ", train_hdf5)
        print("Loading from:", hpav18_train_zip)
        hdf5_writer(
            hpav18_train_zip,
            train_hdf5,
            batch_size,
            chunk_size,
            compression_filter,
            max_limit=num_images,
            append=True,
        )

        # Merge the CSV files
        df = pd.read_csv(csv_path)
        hpav18_df = pd.read_csv(hpav18_csv_path)
        df = pd.concat([df[:num_images], hpav18_df[:num_images]], ignore_index=True)
        df.to_csv(merged_csv, index=False)

    print()
    print("Creating an HDF5 file for the test set")
    print("Loading from:", test_zip)
    print("HDF5 file will be stored at:", test_hdf5)
    hdf5_writer(
        test_zip,
        test_hdf5,
        batch_size,
        chunk_size,
        compression_filter,
        max_limit=num_images,
    )

