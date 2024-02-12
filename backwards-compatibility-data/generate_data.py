import os
import shutil
import numpy as np

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import load_fvecs

# def create_sift_micro():
#     '''
#     Create a smaller version of the base SIFT 10K dataset (http://corpus-texmex.irisa.fr). You 
#     don't need to run this again, but it's saved here just in case. To query an index built with 
#     this data just select vectors from this file as the query vectors.
#     '''
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     base_uri = os.path.join(script_dir, "..", "apis", "python", "test", "data", "siftsmall", "siftsmall_base.fvecs")
#     write_fvecs(os.path.join(script_dir, "siftmicro_base.fvecs"), load_fvecs(base_uri)[:100])

def generate_release_data(version):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the new release directory.
    release_dir = os.path.join(script_dir, "data", version) 
    shutil.rmtree(release_dir, ignore_errors=True)
    os.makedirs(release_dir, exist_ok=True)

    # Get the data we'll use to generate the index.
    base_uri = os.path.join(script_dir, "siftmicro_base.fvecs")
    base = load_fvecs(base_uri)
    indices = [0, 3, 4, 8, 10, 19, 28, 31, 39, 40, 41, 47, 49, 50, 56, 64, 68, 70, 71, 79, 82, 89, 90, 94]
    queries = base[indices]

    # Generate each index and query to make sure it works before we write it.
    index_types = ["FLAT", "IVF_FLAT"]
    data_types = ["float32", "uint8"]
    source_types = ["F32BIN", "U8BIN"]
    for index_type in index_types:
        for data_type, source_type in zip(data_types, source_types):
            numpy_source_data = base.astype(data_type)
            numpy_source_data_uri = os.path.join(script_dir, "numpy_source_data.bin")
            with open(numpy_source_data_uri, "wb") as f:
                np.array(numpy_source_data.shape, dtype="uint32").tofile(f)
                numpy_source_data.tofile(f)

            index_uri = f"{release_dir}/{index_type.lower()}_{data_type}"
            print(f"Creating index at {index_uri}")
            index = ingest(
                index_type=index_type, 
                array_uri=index_uri,
                source_type=source_type,
                source_uri=numpy_source_data_uri,
            )
            result = index.query(queries, k=1)
            assert indices == result.flatten().tolist()

            os.remove(numpy_source_data_uri)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("version", help="The name of the of the TileDB-Vector-Search version which we are creating indices for.")
    args = p.parse_args()
    print(f"Building indexes for version {args.version}")
    generate_release_data(args.version)