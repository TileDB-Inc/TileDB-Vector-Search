# Used to benchmark ingestion and querying running locally. First downloads SIFT and then
# benchmarks ingestion and querying.
#
# To run:
# - ~/repo/TileDB-Vector-Search pip install ".[benchmarks]"
# - ~/repo/TileDB-Vector-Search python apis/python/test/local-benchmarks.py

import logging
import os
import tarfile
import time
import urllib.request
from datetime import datetime
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
from common import accuracy
from common import get_groundtruth_ivec

import tiledb
from tiledb.vector_search.index import Index
from tiledb.vector_search.ingestion import TrainingSamplingPolicy
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.ivf_pq_index import IVFPQIndex
from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.vamana_index import VamanaIndex


class RemoteURIType(Enum):
    LOCAL = 1
    TILEDB = 2
    AWS = 3


## Settings
REMOTE_URI_TYPE = RemoteURIType.LOCAL
USE_SIFT_SMALL = True

# Use headless mode for matplotlib.
matplotlib.use("Agg")

SIFT_URI = (
    "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    if USE_SIFT_SMALL
    else "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
)
SIFT_FOLDER_NAME = "siftsmall" if USE_SIFT_SMALL else "sift"

TEMP_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(TEMP_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(TEMP_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, "logs.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SIFT_DOWNLOAD_PATH = os.path.join(
    TEMP_DIR, "siftsmall.tar.gz" if USE_SIFT_SMALL else "sift.tar.gz"
)
SIFT_BASE_PATH = os.path.join(
    TEMP_DIR,
    SIFT_FOLDER_NAME,
    "siftsmall_base.fvecs" if USE_SIFT_SMALL else "sift_base.fvecs",
)
SIFT_QUERIES_PATH = os.path.join(
    TEMP_DIR,
    SIFT_FOLDER_NAME,
    "siftsmall_query.fvecs" if USE_SIFT_SMALL else "sift_query.fvecs",
)
SIFT_GROUNDTRUTH_PATH = os.path.join(
    TEMP_DIR,
    SIFT_FOLDER_NAME,
    "siftsmall_groundtruth.ivecs" if USE_SIFT_SMALL else "sift_groundtruth.ivecs",
)


def sift_string():
    return "(SIFT 10K)" if USE_SIFT_SMALL else "(SIFT 1M)"


class TimerMode(Enum):
    INGESTION = "ingestion"
    QUERY = "query"


class Timer:
    def __init__(self, name):
        self.name = name
        self.current_timers = {}

        self.keyToTimes = {}
        self.tagToAccuracies = {}

    def start(self, tag, mode):
        key = f"{tag}_{mode.value}"
        if key in self.current_timers:
            raise ValueError(f"Timer {tag} already started.")
        self.current_timers[key] = time.time()

    def stop(self, tag, mode):
        key = f"{tag}_{mode.value}"
        if key not in self.current_timers:
            raise ValueError(f"Timer {tag} not started.")
        elapsed = time.time() - self.current_timers[key]
        self.current_timers.pop(key)

        if key not in self.keyToTimes:
            self.keyToTimes[key] = []
        self.keyToTimes[key].append(elapsed)
        return elapsed

    def accuracy(self, tag, acc):
        if tag not in self.tagToAccuracies:
            self.tagToAccuracies[tag] = []
        self.tagToAccuracies[tag].append(acc)
        return acc

    def _summarize_data(self):
        summary = {}
        for key, intervals in self.keyToTimes.items():
            tag, mode = key.rsplit("_", 1)
            if tag not in summary:
                summary[tag] = {
                    "ingestion": {"total_time": 0, "count": 0, "times": []},
                    "query": {
                        "total_time": 0,
                        "count": 0,
                        "accuracies": [],
                        "times": [],
                    },
                }
            total_time = sum(intervals)
            count = len(intervals)
            if mode == "ingestion":
                summary[tag]["ingestion"]["total_time"] += total_time
                summary[tag]["ingestion"]["count"] += count
                summary[tag]["ingestion"]["times"] = intervals
            elif mode == "query":
                summary[tag]["query"]["total_time"] += total_time
                summary[tag]["query"]["count"] += count
                summary[tag]["query"]["times"] = intervals

        for tag, accuracies in self.tagToAccuracies.items():
            if tag in summary:
                summary[tag]["query"]["accuracies"] = accuracies

        return summary

    def _summary_string(self):
        summary = self._summarize_data()
        summary_str = f"Timer: {self.name}\n"
        for tag, data in summary.items():
            summary_str += f"{tag}\n"
            if "ingestion" in data:
                summary_str += f"  Ingestion (count: {data['ingestion']['count']}):\n"
                summary_str += f"    Average Time: {data['ingestion']['total_time'] / data['ingestion']['count']:.4f} seconds\n"
            if "query" in data:
                summary_str += f"  Query (count: {data['query']['count']}):\n"
                summary_str += f"    Average Time: {data['query']['total_time'] / data['query']['count']:.4f} seconds\n"
                if data["query"]["accuracies"]:
                    summary_str += f"    Average Accuracy: {sum(data['query']['accuracies']) / len(data['query']['accuracies']):.4f}\n"
            summary_str += "\n"
        return summary_str

    def add_data_to_ingestion_time_vs_average_query_accuracy(self, marker="o"):
        summary = self._summarize_data()

        for tag, data in summary.items():
            ingestion_times = []
            average_accuracy = sum(data["query"]["accuracies"]) / len(
                data["query"]["accuracies"]
            )
            for i in range(data["ingestion"]["count"]):
                ingestion_times.append(
                    (data["ingestion"]["times"][i], average_accuracy)
                )
            x, y = zip(*ingestion_times)
            plt.scatter(y, x, marker=marker, label=tag)

    def add_data_to_query_time_vs_accuracy(self, marker="o"):
        summary = self._summarize_data()

        for tag, data in summary.items():
            query_times = []
            for i in range(data["query"]["count"]):
                query_times.append(
                    (data["query"]["times"][i], data["query"]["accuracies"][i])
                )
            x, y = zip(*query_times)
            plt.plot(y, x, marker=marker, label=tag)

    def save_charts(self):
        # Plot ingestion.
        plt.figure(figsize=(20, 12))
        plt.xlabel("Average Query Accuracy")
        plt.ylabel("Time (seconds)")
        plt.title(
            f"{self.name}: Ingestion Time vs Average Query Accuracy {sift_string()}"
        )
        self.add_data_to_ingestion_time_vs_average_query_accuracy()
        plt.legend()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"{self.name}_ingestion_time_vs_accuracy.png")
        )
        plt.close()

        # Plot query.
        plt.figure(figsize=(20, 12))
        plt.xlabel("Accuracy")
        plt.ylabel("Time (seconds)")
        plt.title(f"{self.name}: Query Time vs Accuracy {sift_string()}")
        self.add_data_to_query_time_vs_accuracy()
        plt.legend()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"{self.name}_query_time_vs_accuracy.png")
        )
        plt.close()

    def save_and_print_results(self):
        summary_string = self._summary_string()
        logger.info(summary_string)

        self.save_charts()


class TimerManager:
    def __init__(self):
        self.timers = []

    def new_timer(self, name):
        timer = Timer(name)
        self.timers.append(timer)
        return timer

    def save_charts(self):
        markers = ["o", "^", "D", "*", "P", "s", "2"]

        # Plot ingestion.
        plt.figure(figsize=(20, 12))
        plt.xlabel("Average Query Accuracy")
        plt.ylabel("Time (seconds)")
        plt.title(f"Ingestion Time vs Average Query Accuracy {sift_string()}")
        for idx, timer in enumerate(self.timers):
            timer.add_data_to_ingestion_time_vs_average_query_accuracy(
                markers[idx % len(markers)]
            )
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "ingestion_time_vs_accuracy.png"))
        plt.close()

        # Plot query.
        plt.figure(figsize=(20, 12))
        plt.xlabel("Accuracy")
        plt.ylabel("Time (seconds)")
        plt.title(f"Query Time vs Accuracy {sift_string()}")
        for idx, timer in enumerate(self.timers):
            timer.add_data_to_query_time_vs_accuracy(markers[idx % len(markers)])
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "query_time_vs_accuracy.png"))
        plt.close()


timer_manager = TimerManager()


def download_and_extract(url, download_path, extract_path):
    if os.path.exists(download_path):
        logger.info(
            f"Skipping download of {url} to {download_path} because it already exists."
        )
    else:
        logger.info(f"Downloading {url} to {download_path}.")
        urllib.request.urlretrieve(url, download_path)
        logger.info("Finished download.")

    logger.info("Extracting files.")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
        logger.info("Finished extracting files.")


config = {}


def get_uri(tag):
    global config
    index_name = f"index_{tag.replace('=', '_')}"
    index_uri = ""
    if REMOTE_URI_TYPE == RemoteURIType.LOCAL:
        index_uri = os.path.join(TEMP_DIR, index_name)
    elif REMOTE_URI_TYPE == RemoteURIType.TILEDB:
        from common import create_cloud_uri
        from common import setUpCloudToken

        setUpCloudToken()
        index_uri = create_cloud_uri(index_name, "local_benchmarks")

        config = tiledb.cloud.Config()
    elif REMOTE_URI_TYPE == RemoteURIType.AWS:
        from common import create_cloud_uri
        from common import setUpCloudToken

        setUpCloudToken()
        index_uri = create_cloud_uri(index_name, "local_benchmarks", True)

        config = {
            "vfs.s3.aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "vfs.s3.aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "vfs.s3.region": os.environ["AWS_REGION"],
        }
    else:
        raise ValueError(f"Invalid REMOTE_URI_TYPE {REMOTE_URI_TYPE}")

    logger.info(f"index_uri: {index_uri}")
    Index.delete_index(index_uri, config)
    return index_uri


def cleanup_uri(index_uri):
    Index.delete_index(index_uri, config)


def benchmark_ivf_flat():
    index_type = "IVF_FLAT"
    timer = timer_manager.new_timer(index_type)

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for partitions in [20, 50, 100, 200]:
        tag = f"{index_type}_partitions={partitions}"
        logger.info(f"Running {tag}")

        index_uri = get_uri(tag)

        timer.start(tag, TimerMode.INGESTION)
        ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=SIFT_BASE_PATH,
            config=config,
            partitions=partitions,
            training_sampling_policy=TrainingSamplingPolicy.RANDOM,
        )
        ingest_time = timer.stop(tag, TimerMode.INGESTION)

        # The index returned by ingest() automatically has memory_budget=1000000 set. Open
        # a fresh index so it's clear what config is being used.
        index = IVFFlatIndex(index_uri, config)

        for nprobe in [1, 2, 3, 4, 5, 10, 20]:
            timer.start(tag, TimerMode.QUERY)
            _, result = index.query(queries, k=k, nprobe=nprobe)
            query_time = timer.stop(tag, TimerMode.QUERY)
            acc = timer.accuracy(tag, accuracy(result, gt_i))
            logger.info(
                f"Finished {tag} with nprobe={nprobe}. Ingestion: {ingest_time:.4f}s. Query: {query_time:.4f}s. Accuracy: {acc:.4f}."
            )

        cleanup_uri(index_uri)

    timer.save_and_print_results()


def benchmark_vamana():
    index_type = "VAMANA"
    timer = timer_manager.new_timer(index_type)

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for l_build in [40]:
        for r_max_degree in [10, 15, 20, 25, 30, 35, 40]:
            tag = f"{index_type}_l_build={l_build}_r_max_degree={r_max_degree}"
            logger.info(f"Running {tag}")

            index_uri = get_uri(tag)

            timer.start(tag, TimerMode.INGESTION)
            ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=SIFT_BASE_PATH,
                config=config,
                l_build=l_build,
                r_max_degree=r_max_degree,
                training_sampling_policy=TrainingSamplingPolicy.RANDOM,
            )
            ingest_time = timer.stop(tag, TimerMode.INGESTION)

            index = VamanaIndex(index_uri, config)

            for l_search in [k, k + 50, k + 100, k + 200, k + 400]:
                timer.start(tag, TimerMode.QUERY)
                _, result = index.query(queries, k=k, l_search=l_search)
                query_time = timer.stop(tag, TimerMode.QUERY)
                acc = timer.accuracy(tag, accuracy(result, gt_i))
                logger.info(
                    f"Finished {tag} with l_search={l_search}. Ingestion: {ingest_time:.4f}s. Query: {query_time:.4f}s. Accuracy: {acc:.4f}."
                )

            cleanup_uri(index_uri)

    timer.save_and_print_results()


def benchmark_ivf_pq():
    index_type = "IVF_PQ"
    timer = timer_manager.new_timer(index_type)

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    dimensions = queries.shape[1]
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for partitions in [200]:
        for num_subspaces in [dimensions / 4]:
            for k_factor in [1, 1.5, 2, 4, 8, 16]:
                tag = f"{index_type}_partitions={partitions}_num_subspaces={num_subspaces}_k_factor={k_factor}"
                logger.info(f"Running {tag}")

                index_uri = get_uri(tag)

                timer.start(tag, TimerMode.INGESTION)
                ingest(
                    index_type=index_type,
                    index_uri=index_uri,
                    source_uri=SIFT_BASE_PATH,
                    config=config,
                    partitions=partitions,
                    training_sampling_policy=TrainingSamplingPolicy.RANDOM,
                    num_subspaces=num_subspaces,
                )
                ingest_time = timer.stop(tag, TimerMode.INGESTION)

                # The index returned by ingest() automatically has memory_budget=1000000 set. Open
                # a fresh index so it's clear what config is being used.
                index = IVFPQIndex(index_uri, config)

                for nprobe in [5, 10, 20, 40, 60]:
                    timer.start(tag, TimerMode.QUERY)
                    _, result = index.query(
                        queries, k=k, nprobe=nprobe, k_factor=k_factor
                    )
                    query_time = timer.stop(tag, TimerMode.QUERY)
                    acc = timer.accuracy(tag, accuracy(result, gt_i))
                    logger.info(
                        f"Finished {tag} with nprobe={nprobe}. Ingestion: {ingest_time:.4f}s. Query: {query_time:.4f}s. Accuracy: {acc:.4f}."
                    )

                cleanup_uri(index_uri)

    timer.save_and_print_results()


def main():
    logger.info(f"Saving results to {RESULTS_DIR}")

    download_and_extract(SIFT_URI, SIFT_DOWNLOAD_PATH, TEMP_DIR)

    benchmark_ivf_flat()
    benchmark_vamana()
    benchmark_ivf_pq()

    timer_manager.save_charts()


main()
