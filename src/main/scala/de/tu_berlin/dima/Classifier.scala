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

  def naiveBayesTrainer(sc: SparkContext, trainingSet : RDD[LabeledPoint]) : NaiveBayesModel= {

    val model = NaiveBayes.train(trainingSet, lambda = 1.0)
    //model.save(sc, "/tmp/genreClassification/model")

    model
  }

  def naiveBayesPredicter(sc: SparkContext, testSet : RDD[LabeledPoint], model: NaiveBayesModel) = {
    //val model = NaiveBayesModel.load(sc, "/tmp/genreClassification/model")

    val predictionAndLabel = testSet.map(p => (model.predict(p.features), p.label))
  }
}
