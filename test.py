from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("RecoMartTest")
    .getOrCreate()
)

print(spark.version)

spark.range(5).show()

spark.stop()