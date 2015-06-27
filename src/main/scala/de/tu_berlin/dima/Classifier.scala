package de.tu_berlin.dima

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by mathieu on 24/06/15.
 */
object Classifier {

  def naiveBayesTrainer(sc: SparkContext, trainingSet : RDD[LabeledPoint]/*, testSet : RDD[LabeledPoint]*/) = {

    val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")

    //val predictionAndLabel = testSet.map(p => (model.predict(p.features), p.label))
    //val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testSet.count()

    model.save(sc, "/tmp/genreClassification/model")
  }

  def naiveBayesPredicter(testSet : RDD) = {
    
  }
}
