# Third Party
import grpc
from os import path
import sys
import os
import timeit
import numpy as np
import csv
import time
from threading import Thread

# Local
import caikit
from caikit.runtime.service_factory import ServicePackageFactory

# Add the runtime/library to the path
sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../../"))
)

# Load configuration for Caikit runtime
CONFIG_PATH = os.getenv("CONFIG_PATH")
caikit.configure(CONFIG_PATH)

# NOTE: The model id needs to be a path to folder.
# NOTE: This is relative path to the models directory
MODEL_ID = os.getenv("MODEL")

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

port = os.getenv('CAIKIT_EMBEDDINGS_PORT') if os.getenv('CAIKIT_EMBEDDINGS_PORT') else 443
host = os.getenv('CAIKIT_EMBEDDINGS_HOST') if os.getenv('CAIKIT_EMBEDDINGS_HOST') else 'localhost'

channel = grpc.insecure_channel(f"{host}:{port}")
client_stub = inference_service.stub_class(channel)

# Test control parameter
ITERATIONS = 100
PRINT_RESPONSE = True
MAX_NEW_TOKENS = 248
REQUESTS_PER_SEC = 2  # ZERO means no mercy, no gap between requests!
INITIAL_WARMUP_CYCLES = int(os.environ.get('INITIAL_WARMUP_CYCLES') or '10')

# Latency Array
latList = [None] * ITERATIONS
threadList = [None] * ITERATIONS

texts_dataset = []

# Load dataset from tsv file   
with open(os.getenv('DATASET')) as f:
    tsv_file = csv.reader(f, delimiter="\t")  
    for line in tsv_file:
        text = line[1]  
        print(text)
        texts_dataset.append(text)
    

if __name__ == '__main__':

    def run():
        request = inference_service.messages.EmbeddingTasksRequest(texts=texts_dataset)
        # Fetch predictions from server (infer)
        response = client_stub.EmbeddingTasksPredict(
            request, metadata=[("mm-model-id", MODEL_ID)]
        )
        # Print response
        print("request completed")

    def getLatency(lat,i):
        s = timeit.default_timer()
        run()
        lat[i] = timeit.default_timer() - s

    tik = timeit.default_timer()
    print(f"Starting warmup with {INITIAL_WARMUP_CYCLES} cycles...")
    # INITIAL_WARMUP_CYCLES
    for i in range(INITIAL_WARMUP_CYCLES):
        run()
    tok = timeit.default_timer()

    print(f"Warmup terminated in {tok - tik} sec.")

    # Start throughput test
    start_time = timeit.default_timer()
    # Spawn threads 
    for i in range(ITERATIONS):
        threadList[i] = Thread(target=getLatency, args=(latList, i))
        threadList[i].start()
        if REQUESTS_PER_SEC > 0:
            time.sleep(1/REQUESTS_PER_SEC)
    # Join threads (wait for them to complete)
    for i in range(ITERATIONS):
        threadList[i].join()
    elapsed = timeit.default_timer() - start_time

    # Some threads may fail and do not return response
    latList = list(filter(lambda x: x is not None, latList))

    lat= np.array(latList)
    print(MODEL_ID)
    print('**** THROUGHPUT TEST REPORT ****')
    print("Total Requests: ", ITERATIONS)
    print("Requests per Sec: ", REQUESTS_PER_SEC)
    print("Requests Served: ", len(latList))
    print("Min Latency:", round(np.min(lat),2), ' Sec')
    print("Max Latency:", round(np.max(lat),2), ' Sec')
    print("25 Percentile:", round(np.percentile(lat, 25),2), ' Sec')
    print("50 Percentile:", round(np.percentile(lat, 50),2), ' Sec')
    print("75 Percentile:", round(np.percentile(lat, 75),2), ' Sec')
    print("90 Percentile:", round(np.percentile(lat, 90),2), ' Sec')
    print("99 Percentile:", round(np.percentile(lat, 99),2), ' Sec')
    print("Average Latency:", round(np.average(lat),2), ' Sec')
    print("Throughput: ", round(len(latList)/elapsed,2), 'Requests/Sec' )
    
