import os
import h5py as h5
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_image(image_dir, name, filters=("red", "green", "blue", "yellow")):
    """Gets a numpy array with the specified filters given the file name.

    Arguments:
        image_dir (str): the directory where the images are stored.
        name (str): the name of the image to retrieve from ``image_dir``. Do not include
            the filter in the name.
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
        path = os.path.join(image_dir, img_name)
        img = Image.open(path)
        img_np = np.asarray(img, dtype=np.uint8)
        img_filters.append(img_np)

    img_np = np.stack(img_filters, axis=-1).squeeze()

    return img_np


def image_batches(
    image_dir, image_names, batch_size=200, filters=("red", "green", "blue", "yellow")
):
    """Generates batches of images given their directory and name (without filter)

    Arguments:
        image_dir (str): the directory where the images are stored.
        image_names (array-like of str): a list of names of images to retrieve from
            ``image_dir``. The names should not contain the filter.
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
        batch.append(get_image(image_dir, name))
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


if __name__ == "__main__":
    # Parameters
    root_dir = "../../dataset"
    hdf5_dir = "../../dataset"
    filters = ("red", "green", "blue", "yellow")
    batch_size = 200
    num_chunk_items = 200

    # Just for testing; set to None to store all images in hdf5
    num_images = None

    # Paths
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    csv_path = os.path.join(root_dir, "train.csv")
    train_hdf5 = os.path.join(hdf5_dir, "train.hdf5")
    test_hdf5 = os.path.join(hdf5_dir, "test.hdf5")

    # Get the training image names from the CSV file
    df = pd.read_csv(csv_path)
    train_names = df["Id"].values
    train_names = train_names[:num_images]

    # Get the test images from the filesystem, remove the filters from the names and
    # remove the duplicated names
    test_names = sorted(os.listdir(test_dir))
    test_names = [name.rsplit("_", 1)[0] for name in test_names]
    test_names = list(sorted(set(test_names)))
    test_names = test_names[:num_images]

    # All relevant information for the training and test set compiled in a tuple
    train_info = (train_dir, train_names, train_hdf5)
    test_info = (test_dir, test_names, test_hdf5)

    for image_dir, image_names, hdf5_path in (train_info, test_info):
        print("Image directory:", image_dir)
        print("Number of images:", len(image_names))
        print("HDF5 file will be saved at:", hdf5_path)
        with tqdm(image_names) as pbar:
            # Iterate over the batches and add them to the hdf5 file
            batch_loader = image_batches(image_dir, image_names, batch_size=batch_size)
            for idx, (names, images) in enumerate(batch_loader):
                if idx == 0:
                    # On the first iteration create the file and two datasets. One
                    # contains the images and the other their respective names
                    with h5.File(hdf5_path, "w") as f:
                        f.create_dataset(
                            "images",
                            data=images,
                            maxshape=(len(image_names), *images.shape[1:]),
                            chunks=(num_chunk_items, *images.shape[1:]),
                        )
                        f.create_dataset(
                            "names",
                            data=names,
                            maxshape=(len(image_names),),
                            chunks=(num_chunk_items,),
                        )
                else:
                    # Append the new images and names to the existing hdf5 file
                    with h5.File(hdf5_path, "a") as f:
                        f["images"].resize(f["images"].shape[0] + len(images), axis=0)
                        f["images"][-images.shape[0] :] = images

                        f["names"].resize(f["names"].shape[0] + len(names), axis=0)
                        f["names"][-len(names) :] = names

                pbar.update(len(images))

            print()
