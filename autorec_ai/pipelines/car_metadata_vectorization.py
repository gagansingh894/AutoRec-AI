from autorec_ai.preprocessing.metadata import Vectorizer

if __name__ == '__main__':
    car_metadata_vectorizer = Vectorizer(qdrant_grpc_host='0.0.0.0', qdrant_grpc_port=6334)
    car_metadata_vectorizer.vectorize()