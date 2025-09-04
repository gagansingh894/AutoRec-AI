from collections import defaultdict
from typing import Tuple, Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client import models

from autorec_ai.utils.config import METADATA_PATH, QDRANT_COLLECTION_NAME
from autorec_ai.utils.logger import logger


class Vectorizer:
    """
    A class for transforming structured metadata into vector representations and storing
    them in a Qdrant vector database.

    This class reads a metadata CSV file, applies one-hot encoding to selected categorical
    columns, extracts numerical features, and constructs vector embeddings alongside
    payloads (e.g., make, model, year). The resulting vectors and payloads are uploaded
    into a Qdrant collection for similarity search and recommendations.
    """

    def __init__(self, qdrant_grpc_host: str, qdrant_grpc_port: int):
        """
        Initialize the Vectorizer with a Qdrant gRPC client.

        Args:
            qdrant_grpc_host (str): Host address of the Qdrant server.
            qdrant_grpc_port (int): Port of the Qdrant gRPC server.
        """
        self.qdrant = QdrantClient(
            host=qdrant_grpc_host,
            port=qdrant_grpc_port,
            prefer_grpc=True
        )
        self._logger = logger.bind(component='preprocessing.metadata.Vectorizer')

        # Columns used for numerical feature extraction
        self._want_cols = [
            'mpg_city', 'mpg_highway', 'horsepower', 'torque',
            'weight', 'length', 'width', 'height', 'wheelbase'
        ]

        # Categorical columns that will be one-hot encoded
        self._one_hot_encode_cols = ['num_doors', 'body_style', 'driver_type']

        # Metadata columns to be preserved as payloads in Qdrant
        self._payload_cols = ['make', 'model', 'year']

    def vectorize(self):
        """
        Main entrypoint for the vectorization process.

        - Reads metadata CSV
        - One-hot encodes categorical columns
        - Converts rows into vectors + payloads
        - Uploads them to Qdrant
        """
        df = pd.read_csv(f'{METADATA_PATH}/metadata.csv')
        df = self._one_hot_encode(df)
        self._vectorize(df)

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns and append them
        to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded categorical features.
        """
        one_hot_encoded_dfs = []
        for col in self._want_cols:
            one_hot_encoded_dfs.append(pd.get_dummies(df[col], prefix=col, dtype=int))

        one_hot_encoded_df = pd.concat(one_hot_encoded_dfs, axis=1)

        return pd.concat([df, one_hot_encoded_df], axis=1).drop(
            columns=['num_doors', 'body_style', 'driver_type'], axis=1
        )

    def _vectorize(self, df: pd.DataFrame):
        """
        Converts the processed DataFrame into vectors and payloads,
        creates the Qdrant collection (if not exists), and uploads the points.

        Args:
            df (pd.DataFrame): Processed DataFrame with one-hot encoded and numerical features.
        """
        vector_size = df.shape[1] - len(self._payload_cols)
        print(vector_size)

        # Create collection if it doesn't exist
        if not self.qdrant.collection_exists(QDRANT_COLLECTION_NAME):
            self.qdrant.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

        vectors, payloads = self._get_vectors_payloads(df)

        # BUG: this should use len(vectors), not vector_size
        ids = list(range(len(vectors)))

        print(len(ids), len(vectors[0]), len(payloads[0]))

        points = [
            models.PointStruct(id=idx, vector=vector, payload=payload)
            for idx, vector, payload in zip(ids, vectors, payloads)
        ]

        self.qdrant.upload_points(collection_name=QDRANT_COLLECTION_NAME, points=points)

    def _get_vectors_payloads(self, df: pd.DataFrame) -> Tuple[defaultdict, Any]:
        """
        Split the DataFrame into vectors (numerical + one-hot features)
        and payloads (metadata).

        Args:
            df (pd.DataFrame): Processed DataFrame with payload + features.

        Returns:
            Tuple[List[List[float]], List[dict]]:
                - vectors: 2D list of numerical feature vectors.
                - payloads: list of metadata dicts (make, model, year).
        """
        payloads = df[self._payload_cols].to_dict(orient='records')
        df.drop(self._payload_cols, axis=1, inplace=True)
        vectors = df.values.tolist()

        return vectors, payloads