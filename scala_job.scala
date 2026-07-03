// BONUS (for your mentor's Scala requirement): the same ETL logic, in Scala.
// Run with: spark-shell -i scala_job.scala
// (make sure retail_sales_raw.csv is in the same folder you launch spark-shell from)

import org.apache.spark.sql.{SparkSession, functions => F}

val spark = SparkSession.builder()
  .appName("RetailSalesScalaJob")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("retail_sales_raw.csv")

val cleaned = df.na.drop()
  .withColumn("date", F.to_date($"date", "yyyy-MM-dd"))
  .withColumn("year", F.year($"date"))
  .withColumn("month", F.month($"date"))

val agg = cleaned.groupBy("region", "category", "year", "month")
  .agg(
    F.sum("units_sold").alias("total_units"),
    F.sum("revenue").alias("total_revenue")
  )
  .orderBy("year", "month", "region", "category")

agg.show(20, truncate = false)

agg.coalesce(1).write
  .mode("overwrite")
  .option("header", "true")
  .csv("output/agg_scala")

println("Scala Spark job complete. Check output/agg_scala")