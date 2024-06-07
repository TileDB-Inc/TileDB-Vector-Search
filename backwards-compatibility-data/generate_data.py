import os
import shutil

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.utils import write_fvecs

base_dir = os.path.dirname(os.path.abspath(__file__))


def create_sift_micro():
    """
    Create a smaller version of the base SIFT 10K dataset (http://corpus-texmex.irisa.fr). You
    don't need to run this again, but it's saved here just in case. To query an index built with
    this data just select vectors from this file as the query vectors.
    """
    base_uri = os.path.join(
        base_dir,
        "..",
        "external",
        "test_data",
        "files",
        "siftsmall",
        "input_vectors.fvecs",
    )
    write_fvecs(
        os.path.join(base_dir, "siftmicro_base.fvecs"), load_fvecs(base_uri)[:100]
    )


def generate_indexes(version):
    # Create the new release directory.
    index_dir = os.path.join(base_dir, "data", version)
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir, exist_ok=True)

    # Get the data we'll use to generate the index.
    base_uri = os.path.join(base_dir, "siftmicro_base.fvecs")
    base = load_fvecs(base_uri)
    indices = [
        0,
        3,
        4,
        8,
        10,
        19,
        28,
        31,
        39,
        40,
        41,
        47,
        49,
        50,
        56,
        64,
        68,
        70,
        71,
        79,
        82,
        89,
        90,
        94,
    ]
    queries = base[indices]

    # Generate each index and query to make sure it works before we write it.
    index_types = ["FLAT", "IVF_FLAT", "VAMANA"]
    data_types = ["float32", "uint8"]
    for index_type in index_types:
        for data_type in data_types:
            index_uri = f"{index_dir}/{index_type.lower()}_{data_type}"
            index = ingest(
                index_type=index_type,
                index_uri=index_uri,
                input_vectors=base.astype(data_type),
            )

            result_d, result_i = index.query(queries, k=1)
            assert indices == result_i.flatten().tolist()
            assert result_d.flatten().tolist() == [0 for _ in range(len(indices))]


def check_should_upload_indexes(version) -> bool:
    """
    Returns True if the minor version of the version string is greater than the minor version of the last version uploaded. When we run on CI we only want to upload data when the minor version changes. Examples:
    - We have 0.1.0, 0.1.1. We get version=0.1.2. In this case we return False.
    - We have 0.1.0, 0.1.1. We get version=0.2.0. In this case we return True.
    - We have 0.1.0, 0.1.1. We get version=0.2.9. In this case we return True.
    """
    split_version = args.version.split(".")
    minor_version = split_version[1] if len(split_version) >= 2 else None
    if minor_version is None:
        return False

    data_dir = os.path.join(base_dir, "data")
    for folder in os.listdir(data_dir):
        split_folder = folder.split(".")
        folder_minor_version = split_folder[1] if len(split_folder) >= 2 else None
        if folder_minor_version == minor_version:
            return False

    return True


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "version",
        help="The name of the of the TileDB-Vector-Search version which we are creating indices for.",
    )
    args = p.parse_args()

    should_upload_indexes = check_should_upload_indexes(args.version)

    generate_indexes(args.version)

    print("true" if should_upload_indexes else "false")
