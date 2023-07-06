package classificationTask

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, input_file_name, regexp_extract, udf}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.regression.LabeledPoint
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

    metrics.confusionMatrix.rowIter
    // iterate over the elements in the confusion matrix and print them out
    for (i <- 0 until metrics.confusionMatrix.numRows) {
      for (j <- 0 until metrics.confusionMatrix.numCols) {
        println(s"Element ($i, $j) = ${metrics.confusionMatrix(i, j)}")
      }
    }

    //val predictionAndLabels = Seq((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
    //val predictionAndLabelsRdd = spark.sparkContext.parallelize(predictionAndLabels)
    //val metrics = new BinaryClassificationMetrics(predictionAndLabelsRdd)

    labels.foreach { l =>
      val modifiedPredictionAndLabelsRdd = predictionAndLabelsEvaluation.map { case (prediction, lb) =>
        // Operate on the value of prediction here
        if(prediction == lb && lb == l){
          (1, 1)
        }else if(prediction == lb && lb != l){
          (0,0)
        }else if(prediction != lb  ){
          ()
        }
      }
    }


    spark.stop()

  }
}
