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
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

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
      .setNumFolds(5)

    val cvModel = cv.fit(personality)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val rfModel = bestModel.stages.last.asInstanceOf[RandomForestClassificationModel]
    cvModel.transform(testData).show()

    val predictionsAndLabels = bestModel.transform(testData).select("prediction", label)
    predictionsAndLabels.show()

    val accuracy = evaluator.evaluate(predictionsAndLabels)

    val evaluator2 = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
      .setLabelCol(label)
      .setPredictionCol("prediction")
    val weightedPrecision = evaluator2.evaluate(predictionsAndLabels)

    val evaluator3 = new MulticlassClassificationEvaluator()
      .setMetricName("weightedRecall")
      .setLabelCol(label)
      .setPredictionCol("prediction")
    val weightedRecall = evaluator3.evaluate(predictionsAndLabels)

    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol(label)
      .setPredictionCol("prediction")
    val f1 = evaluator4.evaluate(predictionsAndLabels)


    val evaluator5 = new BinaryClassificationEvaluator()
      .setLabelCol(label)
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val areaUnderROC = evaluator5.evaluate(predictionsAndLabels)


    // Extract the scores and labels from the predictions
    val scoresAndLabels = predictionsAndLabels.select("probability", label)
      .rdd.map(row => (row.getAs[org.apache.spark.ml.linalg.Vector](0)(1), row.getDouble(1)))

    // Instantiate a BinaryClassificationMetrics object
    val metrics = new BinaryClassificationMetrics(scoresAndLabels)

    // Compute the ROC curve for each label
    val rocCurves = metrics.roc()

    // Print the ROC curve for each label
    for (i <- 0 until 3) {
      //todo:println(s"ROC curve for label $i: ${rocCurves(i).collect().toList}")
    }

    println(s"Accuracy: $accuracy")
    println(s"weightedPrecision: $weightedPrecision")
    println(s"weightedRecall: $weightedRecall")
    println(s"f1: $f1")
    println(s"areaUnderROC: $areaUnderROC")
    println(s"Number of trees: $rfModel.getNumTrees")
    println(s"Max depth: $rfModel.getMaxDepth")

    //bestModel.write.overwrite().save("model")
    spark.stop()

  }
}
