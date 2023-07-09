package classificationTask

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object DecisionTree {
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
      .load("dataset/marketing_campaign_cleaned_classification_rft.csv")

    val label = "Income"

    val label_0 = personality.filter(col(label) === 0)
    val label_1 = personality.filter(col(label) === 1)
    val label_2 = personality.filter(col(label) === 2)
    val smallestClassSize = Seq(label_0.count(), label_1.count(), label_2.count()).min
    val balancedData = label_0.limit(smallestClassSize.toInt).union(label_1.limit(smallestClassSize.toInt)).union(label_2.limit(smallestClassSize.toInt))
    //balancedData.write.format("csv").option("header",1).option("path", "dataset/marketing_campaign_balanced_classification.csv").save()


    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = balancedData.randomSplit(Array(0.7, 0.3), seed = 1234L)


    val assembler = new VectorAssembler()
      .setInputCols(balancedData.columns.filter(_ != label))
      .setOutputCol("features")

    balancedData.show()

    // Train a NaiveBayes model.
    val dt = new DecisionTreeClassifier()
      .setFeaturesCol("features")
      .setLabelCol(label)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, dt))

    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 10))
      .addGrid(dt.impurity, Array("gini", "entropy"))
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
    cvModel.transform(testData).show()

    val predictionsAndLabels = bestModel.transform(testData).select("prediction", label)
    predictionsAndLabels.show()


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
