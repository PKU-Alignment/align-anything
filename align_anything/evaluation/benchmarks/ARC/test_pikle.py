import os
import pickle
from pprint import pprint
from align_anything.evaluation.data_type import InferenceOutput

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
    
def main():
    cache_path = ".cache"
    files = os.listdir(cache_path)
    InferenceOutputs = []
    for file in files:
        if file.endswith(".pkl"):
            InferenceOutputs.extend(load_pickle(os.path.join(cache_path, file)))
    pprint([InferenceOutputs[i].response for i in range(10)])
    pprint(len(InferenceOutputs))
    
if __name__=="__main__":
    main()
