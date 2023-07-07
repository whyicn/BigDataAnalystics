package classificationTask

import org.apache.hadoop.shaded.org.eclipse.jetty.websocket.common.frames.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, input_file_name, regexp_extract, udf, when}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.sparkproject.dmg.pmml.ConfusionMatrix

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

    val label = "Income"

    val label_0 = personality.filter(col(label) === 0)
    val label_1 = personality.filter(col(label) === 1)
    val label_2 = personality.filter(col(label) === 2)
    val smallestClassSize = Seq(label_0.count(), label_1.count(), label_2.count()).min
    val balancedData = label_0.limit(smallestClassSize.toInt).union(label_1.limit(smallestClassSize.toInt)).union(label_2.limit(smallestClassSize.toInt))
    //balancedData.write.format("csv").option("header",1).option("path", "dataset/marketing_campaign_balanced_classification.csv").save()

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = balancedData.randomSplit(Array(0.7, 0.3), seed = 1234L)

    personality.show()


    val assembler = new VectorAssembler()
      .setInputCols(balancedData.columns.filter(_ != label))
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

    val cvModel = cv.fit(balancedData)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val rfModel = bestModel.stages.last.asInstanceOf[RandomForestClassificationModel]
    cvModel.transform(testData).show()

    val predictionsAndLabels = bestModel.transform(testData).select("prediction", label)
    predictionsAndLabels.show()

    println(s"Number of trees: $rfModel.getNumTrees")
    println(s"Max depth: $rfModel.getMaxDepth")


    val predictionAndLabelsEvaluation = predictionsAndLabels.select(col("prediction"), col(label).cast("Double"))
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabelsEvaluation)
    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracys = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracys")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // True positive rate by label
    labels.foreach { l =>
      println(s"TPR($l) = " + metrics.truePositiveRate(l))
    }


    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }


    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

    //val predictionAndLabels = Seq((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
    //val predictionAndLabelsRdd = spark.sparkContext.parallelize(predictionAndLabels)

    spark.stop()

  }
}
