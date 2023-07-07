package clusterTask

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
object K_means {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("scene")
      .master("local[*]")
      //      .config("spark.sql.warehouse.dir", "images/train")
      .getOrCreate()

    // Load the labels from CSV file
    val personality = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("dataset/marketing_campaign_cleaned_cluster_balance_1.csv")

    val label = "Income"

    val assembler = new VectorAssembler()
      .setInputCols(personality.columns.filter(_ != label))
      .setOutputCol("features")

    val data = assembler.transform(personality).select("features")

    // Trains a bisecting k-means model.
    val bkm = new BisectingKMeans().setK(3).setSeed(123)
    val model = bkm.fit(data)

    // Make predictions
    val predictions = model.transform(data)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    val centers = model.clusterCenters
    centers.foreach(println)


  }
}
