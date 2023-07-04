package classificationTask

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, input_file_name, regexp_extract, udf}

object RandomForestTree {
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
      .load("dataset/marketing_campaign.csv")

    personality.show()

    val label = "Response"

    val categoricalColumns = Array("Marital_Status", "Education")

    val categoricalIndexers = categoricalColumns.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_indexed")
    }

    val assembler = new VectorAssembler()
      .setInputCols(categoricalColumns.map(_ + "_indexed") ++ Array(label))
      .setOutputCol("features")

    val Array(trainingData, testData) = personality.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier()
      .setLabelCol("Response")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(categoricalIndexers ++ Array(assembler, rf))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    println("==================")
    predictions.show()
    println("==================")

    val predictionAndLabels = model.transform(personality).select("prediction", label).rdd.map(row => (row.getDouble(0), row.getInt(1).toDouble))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println(s"Area under ROC curve = ${metrics.areaUnderROC()}")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    println(s"Weighted precision = ${evaluator.evaluate(predictions)}")

    evaluator.setMetricName("weightedRecall")
    println(s"Weighted recall = ${evaluator.evaluate(predictions)}")

    evaluator.setMetricName("f1")
    println(s"F1 score = ${evaluator.evaluate(predictions)}")


    spark.stop()

  }
}
