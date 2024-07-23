# Used to benchmark ingestion and querying running locally. First downloads SIFT and then 
# benchmarks ingestion and querying.
#
# To run:
# - ~/repo/TileDB-Vector-Search pip install .
# - ~/repo/TileDB-Vector-Search python apis/python/test/local-benchmarks.py

import os
import shutil
import tarfile
import time
import urllib.request
from enum import Enum

from common import accuracy
from common import get_groundtruth_ivec

from tiledb.vector_search.ingestion import TrainingSamplingPolicy
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import load_fvecs

SIFT_URI = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

TEMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

SIFT_BASE_PATH = os.path.join(TEMP_DIR, "sift", "sift_base.fvecs")
SIFT_QUERIES_PATH = os.path.join(TEMP_DIR, "sift", "sift_query.fvecs")
SIFT_GROUNDTRUTH_PATH = os.path.join(TEMP_DIR, "sift", "sift_groundtruth.ivecs")

class TimerMode(Enum):
    INGESTION = 'ingestion'
    QUERY = 'query'

class Timer:
    def __init__(self):
        self.times = {}
        self.accuracies = {}

    def start(self, tag, mode):
        key = f"{tag}_{mode.value}"
        if key not in self.times:
            self.times[key] = []
        self.times[key].append({'start': time.time(), 'end': None})

    def stop(self, tag, mode):
        key = f"{tag}_{mode.value}"
        if key in self.times and self.times[key][-1]['end'] is None:
            self.times[key][-1]['end'] = time.time()
        else:
            print(f"Warning: Timer for tag '{tag}' and mode '{mode}' was not started or already stopped.")

    def accuracy(self, tag, acc):
        if tag not in self.accuracies:
            self.accuracies[tag] = []
        self.accuracies[tag].append(acc)

    def summarize(self):
        summary = {}
        for key, intervals in self.times.items():
            tag, mode = key.rsplit('_', 1)
            if tag not in summary:
                summary[tag] = {'ingestion': {'total_time': 0, 'count': 0},
                                'query': {'total_time': 0, 'count': 0, 'accuracies': []}}
            total_time = sum(interval['end'] - interval['start'] for interval in intervals if interval['end'] is not None)
            count = len([interval for interval in intervals if interval['end'] is not None])
            if mode == 'ingestion':
                summary[tag]['ingestion']['total_time'] += total_time
                summary[tag]['ingestion']['count'] += count
            elif mode == 'query':
                summary[tag]['query']['total_time'] += total_time
                summary[tag]['query']['count'] += count

        for tag, accuracies in self.accuracies.items():
            if tag in summary:
                summary[tag]['query']['accuracies'] = accuracies

        summary_str = ""
        for tag, data in summary.items():
            summary_str += f"{tag}\n"
            if 'ingestion' in data:
                summary_str += f"  Ingestion (count: {data['ingestion']['count']}):\n"
                summary_str += f"    Average Time: {data['ingestion']['total_time'] / data['ingestion']['count']:.4f} seconds\n"
            if 'query' in data:
                summary_str += f"  Query (count: {data['query']['count']}):\n"
                summary_str += f"    Average Time: {data['query']['total_time'] / data['query']['count']:.4f} seconds\n"
                if data['query']['accuracies']:
                    summary_str += f"    Average Accuracy: {sum(data['query']['accuracies']) / len(data['query']['accuracies']):.4f}\n"
            summary_str += "\n"
        return summary_str

def download_and_extract(url, download_path, extract_path):
  if os.path.exists(download_path):
    print(f"Skipping download of {url} to {download_path} because it already exists.")
  else:
    print(f"Downloading {url} to {download_path}.")
    urllib.request.urlretrieve(url, download_path)
    print("Finished download.")

  print("Extracting files.")
  with tarfile.open(download_path, "r:gz") as tar:
      tar.extractall(path=extract_path)
      print("Finished extracting files.")

def benchmark_ivf_flat():
    index_type = "IVF_FLAT"
    timer = Timer()

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for partitions in [20, 50]:
      tag = f"{index_type}_partitions={partitions}"
      print(f"Running {tag}")

      index_uri = os.path.join(TEMP_DIR, f"index_{index_type}")
      if os.path.exists(index_uri):
          shutil.rmtree(index_uri)

      timer.start(tag, TimerMode.INGESTION)
      index = ingest(
          index_type=index_type,
          index_uri=index_uri,
          source_uri=SIFT_BASE_PATH,
          partitions=partitions,
          training_sampling_policy=TrainingSamplingPolicy.RANDOM,
      )
      timer.stop(tag, TimerMode.INGESTION)

      for nprobe in [5, 10]:
        timer.start(tag, TimerMode.QUERY)
        _, result = index.query(queries, k=k, nprobe=nprobe)
        timer.stop(tag, TimerMode.QUERY)
        timer.accuracy(tag, accuracy(result, gt_i))

    print(timer.summarize())

def benchmark_vamana():
    index_type = "VAMANA"
    timer = Timer()

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for l_build in [5, 10]:
      for r_max_degree in [10]:
        tag = f"{index_type}_l_build={l_build}_r_max_degree={r_max_degree}"
        print(f"Running {tag}")

        index_uri = os.path.join(TEMP_DIR, f"index_{index_type}")
        if os.path.exists(index_uri):
            shutil.rmtree(index_uri)

        timer.start(tag, TimerMode.INGESTION)
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=SIFT_BASE_PATH,
            l_build=l_build,
            r_max_degree=r_max_degree,
            training_sampling_policy=TrainingSamplingPolicy.RANDOM,
        )
        timer.stop(tag, TimerMode.INGESTION)

        for l_search in [k, k + 50, k + 100]:
          timer.start(tag, TimerMode.QUERY)
          _, result = index.query(queries, k=k, l_search=l_search)
          timer.stop(tag, TimerMode.QUERY)
          timer.accuracy(tag, accuracy(result, gt_i))

    print(timer.summarize())

def benchmark_ivf_pq():
    index_type = "IVF_PQ"
    timer = Timer()

    k = 100
    queries = load_fvecs(SIFT_QUERIES_PATH)
    dimensions = queries.shape[1]
    gt_i, gt_d = get_groundtruth_ivec(SIFT_GROUNDTRUTH_PATH, k=k, nqueries=len(queries))

    for partitions in [20, 50]:
      for num_subspaces in [dimensions / 2, dimensions / 4]:
        tag = f"{index_type}_partitions={partitions}_num_subspaces={num_subspaces}"
        print(f"Running {tag}")

        index_uri = os.path.join(TEMP_DIR, f"index_{index_type}")
        if os.path.exists(index_uri):
            shutil.rmtree(index_uri)

        timer.start(tag, TimerMode.INGESTION)
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=SIFT_BASE_PATH,
            partitions=partitions,
            training_sampling_policy=TrainingSamplingPolicy.RANDOM,
            num_subspaces=num_subspaces
        )
        timer.stop(tag, TimerMode.INGESTION)

        for nprobe in [5, 10]:
          timer.start(tag, TimerMode.QUERY)
          _, result = index.query(queries, k=k, nprobe=nprobe)
          timer.stop(tag, TimerMode.QUERY)
          timer.accuracy(tag, accuracy(result, gt_i))

    print(timer.summarize())

def main():
  download_and_extract(SIFT_URI, os.path.join(TEMP_DIR, "sift.tar.gz"), TEMP_DIR)

  # benchmark_ivf_flat()
  # benchmark_vamana()
  benchmark_ivf_pq()

main()
