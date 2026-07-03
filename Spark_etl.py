"""
STEP 2: PySpark job - clean and aggregate the raw sales data.
This is the "data engineering" part of the project (uses your PySpark skill).
Run: python spark_etl.py
"""
import os


os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("RetailSalesETL")
    .master("local[*]")
    .getOrCreate()
)

# 1. Read raw CSV
df = spark.read.csv("retail_sales_raw.csv", header=True, inferSchema=True)

# 2. Clean
df = df.dropna()
df = df.withColumn("date", F.to_date("date", "yyyy-MM-dd"))
df = df.withColumn("year", F.year("date"))
df = df.withColumn("month", F.month("date"))

# 3. Aggregate: sales by region + category + month (main dataset the agent will query)
agg_region_category_month = (
    df.groupBy("region", "category", "year", "month")
    .agg(
        F.sum("units_sold").alias("total_units"),
        F.sum("revenue").alias("total_revenue"),
    )
    .orderBy("year", "month", "region", "category")
)

# 4. Aggregate: overall monthly totals
agg_monthly = (
    df.groupBy("year", "month")
    .agg(
        F.sum("units_sold").alias("total_units"),
        F.sum("revenue").alias("total_revenue"),
    )
    .orderBy("year", "month")
)

agg_region_category_month.show(10, truncate=False)

# 5. Save as CSV (coalesce(1) = single output file, easy to read later with pandas)
agg_region_category_month.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("output/agg_region_category_month")

agg_monthly.coalesce(1).write.mode("overwrite").option("header", True).csv(
    "output/agg_monthly"
)

print("ETL complete. Check the 'output' folder.")
spark.stop()