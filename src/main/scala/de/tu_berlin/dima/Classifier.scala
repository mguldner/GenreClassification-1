package de.tu_berlin.dima

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by mathieu on 24/06/15.
 */
object Classifier {

  def naiveBayesTrainer(sc: SparkContext, trainingSet : RDD[LabeledPoint], lambda: Double) = {

    val model = NaiveBayes.train(trainingSet, lambda = lambda)
    model.save(sc, "file:///tmp/genreClassification/model")
  }

  def naiveBayesPredicter(sc: SparkContext, testSet : RDD[LabeledPoint]) : RDD[(Double, Double)] = {
    val model = NaiveBayesModel.load(sc, "file:///tmp/genreClassification/model")

    val predictionAndLabel : RDD[(Double, Double)] = testSet.map(p => (model.predict(p.features), p.label ))

    predictionAndLabel
  }
}
