import pickle


def save_pipeline(pipeline, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)