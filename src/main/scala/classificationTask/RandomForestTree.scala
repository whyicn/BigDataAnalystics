package classificationTask

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
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
      .load("dataset/marketing_campaign_cleaned_classification_3.csv")
    val Array(trainingData, testData) = personality.randomSplit(Array(0.7, 0.3))

    personality.show()

    val label = "Income"
    val assembler = new VectorAssembler()
      .setInputCols(personality.columns.filter(_ != label))
      .setOutputCol("features")

    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setNumTrees(10)
      .setMaxDepth(5)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, rf))

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20))
      .addGrid(rf.maxDepth, Array(5, 10))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(personality)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val rfModel = bestModel.stages.last.asInstanceOf[RandomForestClassificationModel]

    println(s"Accuracy: ${evaluator.evaluate(cvModel.transform(testData))}")
    println(s"Number of trees: ${rfModel.getNumTrees}")
    println(s"Max depth: ${rfModel.getMaxDepth}")

    spark.stop()

  }
}
