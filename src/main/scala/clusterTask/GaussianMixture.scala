package clusterTask

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
object GaussianMixture {
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
      .load("dataset/marketing_campaign_cleaned_cluster_balance.csv")

    val label = "Income"

    val assembler = new VectorAssembler()
      .setInputCols(personality.columns.filter(_ != label))
      .setOutputCol("features")

    val data = assembler.transform(personality).select("features")



      val kmeans = new KMeans()
        .setK(2)
        .setSeed(1L)

      val model = kmeans.fit(data)
      val predictions = model.transform(data)

      val evaluator = new ClusteringEvaluator()
        .setDistanceMeasure("squaredEuclidean")
        .setPredictionCol("prediction")

      val silhouette = evaluator.evaluate(predictions)

      println(s"Silhouette with squared euclidean distance = $silhouette")

      val dbIndex = evaluator.evaluate(predictions)
      println(s"Davies-Bouldin index = $dbIndex")

      val chIndex = evaluator.evaluate(predictions)
      println(s"Calinski-Harabasz index = $chIndex")


  }
}
