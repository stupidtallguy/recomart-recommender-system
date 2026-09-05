from __future__ import annotations

from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F


class ALSRecommender:
    """
    Implicit-feedback collaborative filtering using PySpark ALS.

    Interaction strength:
        number of historical purchases for a user-product pair.

    Recommendations are precomputed after training so that
    evaluation uses the same recommend(user_id, k) interface
    as the other RecoMart models.
    """

    def __init__(
        self,
        rank: int = 32,
        max_iter: int = 10,
        reg_param: float = 0.05,
        alpha: float = 20.0,
        precompute_k: int = 20,
        seed: int = 42,
    ) -> None:

        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.alpha = alpha
        self.precompute_k = precompute_k
        self.seed = seed

        self.model = None
        self.recommendations = None

        self.training_seconds = None
        self.recommendation_seconds = None
        self.unique_user_item_pairs = None


    def fit(
        self,
        spark,
        train_path: Path,
    ) -> "ALSRecommender":

        print(
            "Reading training interactions with Spark..."
        )

        interactions = (
            spark.read
            .parquet(str(train_path))
            .select(
                F.col("user_id")
                .cast("int")
                .alias("user_id"),

                F.col("product_id")
                .cast("int")
                .alias("product_id"),
            )
        )

        print(
            "Building implicit interaction strengths..."
        )

        # First ALS signal:
        #
        # user-product purchase frequency.
        #
        # Example:
        #
        # user 10 + banana -> 8 purchases
        # user 10 + milk   -> 4 purchases
        interaction_strength = (
            interactions
            .groupBy(
                "user_id",
                "product_id",
            )
            .agg(
                F.count("*")
                .cast("float")
                .alias("strength")
            )
            .persist(
                StorageLevel.MEMORY_AND_DISK
            )
        )

        self.unique_user_item_pairs = (
            interaction_strength.count()
        )

        print(
            "Unique user-product pairs: "
            f"{self.unique_user_item_pairs:,}"
        )

        print(
            "\nTraining implicit ALS..."
        )

        print(
            f"rank={self.rank}, "
            f"maxIter={self.max_iter}, "
            f"regParam={self.reg_param}, "
            f"alpha={self.alpha}"
        )

        als = ALS(
            userCol="user_id",
            itemCol="product_id",
            ratingCol="strength",

            implicitPrefs=True,

            rank=self.rank,
            maxIter=self.max_iter,
            regParam=self.reg_param,
            alpha=self.alpha,

            coldStartStrategy="drop",

            seed=self.seed,
        )

        start = perf_counter()

        self.model = als.fit(
            interaction_strength
        )

        self.training_seconds = (
            perf_counter() - start
        )

        print(
            "\nALS training completed."
        )

        print(
            "Training seconds: "
            f"{self.training_seconds:.2f}"
        )

        print(
            "\nPrecomputing recommendations..."
        )

        start = perf_counter()

        spark_recommendations = (
            self.model
            .recommendForAllUsers(
                self.precompute_k
            )
        )

        # Only ~206K rows are returned here,
        # each containing an array of Top-N recommendations.
        recommendation_rows = (
            spark_recommendations.collect()
        )

        self.recommendations = {}

        for row in recommendation_rows:

            user_id = int(
                row["user_id"]
            )

            products = [
                int(rec["product_id"])
                for rec
                in row["recommendations"]
            ]

            self.recommendations[
                user_id
            ] = products

        self.recommendation_seconds = (
            perf_counter() - start
        )

        print(
            "Users with precomputed recommendations: "
            f"{len(self.recommendations):,}"
        )

        print(
            "Recommendation generation seconds: "
            f"{self.recommendation_seconds:.2f}"
        )

        interaction_strength.unpersist()

        return self


    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:

        if self.recommendations is None:

            raise RuntimeError(
                "ALS model must be fitted first."
            )

        if k > self.precompute_k:

            raise ValueError(
                f"k={k} exceeds precomputed "
                f"limit {self.precompute_k}."
            )

        return (
            self.recommendations
            .get(
                int(user_id),
                [],
            )
            [:k]
        )