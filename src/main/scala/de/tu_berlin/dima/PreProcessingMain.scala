package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
 * Created by oliver on 13.06.15.
 */
object PreProcessingMain {
  //TODO same as in other class with line delim
  def main(args: Array[String]) {

    // input argument flags
    val genresPathFlag = "genres"
    val synopsisPathFlag = "synopses"
    // val lambdaPath = "lambda"

    // set up global parameters
   // if(args.length != 2) throw new RuntimeException("usage: [path-to-genres.list] [path-to-synopses.list] [lambda-value]")
    //val genrePath = args(0)
    //val synopsisPath = args(1)
    //val lambda = args(2)
    val genrePath = "/home/mathieu/Travail/TUBerlin/AIM3/GitHub/aim3/Project/genres.list"
    val synopsisPath = "/home/mathieu/Travail/TUBerlin/AIM3/GitHub/aim3/Project/plot_sample2.list"

    // set up exectution
    val inConf = new Configuration()
    inConf.set("textinputformat.record.delimiter", PreProcessing.synopsis_line_delim)
    val conf = new SparkConf()
      .setAppName("Pre Processing")
      .setMaster("local[4]")
    //.set("spark.driver.allowMultipleContexts", "true")
    val sc = new SparkContext(conf)


    // Creation of the two datasets (trainingSet and testSet)

    val sets = PreProcessing.preProcess(genrePath, synopsisPath, sc)
    val training = sets._1
    val test = sets._2
    training
      .map(m => (m.title, m.year, m.genre))
      .coalesce(1)
      .saveAsTextFile("/tmp/genreClassification/out/trainingSet")
    test
      .map(m => (m.title, m.year, m.genre))
      .coalesce(1)
      .saveAsTextFile("/tmp/genreClassification/out/testSet")

    print("\n\n\n\n\n\n Training?  " + (training.count()==0)+"\n\n\n\n\n\n")
    print("\n\n\n\n\n\n Test ? "+(test.count()==0)+"\n\n\n\n\n\n")
    val trainingSetFitted = PreProcessing.tfidfTransformation(training)
    Classifier.naiveBayesTrainer(sc, trainingSetFitted, 1.0)
    val testSetFitted = PreProcessing.tfidfTransformation(test)
    val prediction : RDD[(Double, Double)] = Classifier.naiveBayesPredicter(sc, testSetFitted)

    // True if the genre for the movie is correct --> (predition, true)
    val predictionIsCorrect = prediction.map(p => {
      (p, p._1 == p._2)
    })

    val accuracy = predictionIsCorrect.filter(p => p._2).count() / predictionIsCorrect.count()
    print("\n\n\n\n\n\nThe general accuracy of the classifier is : " + accuracy+"\n\n\n\n\n\n")

    // By genre, if the genre for the movie is good --> (prediction, true)
    val accuracyByGenre : RDD[(Double, Int)] =
      predictionIsCorrect
        .groupBy((p: ((Double, Double), Boolean)) => p._1._2)
        .map(pByGenre => {
        //val numberRight = pByGenre._2.count(p => p._2)
        //val totalNumber = pByGenre._2.count(p=> (p._2 || !p._2)
        (pByGenre._1, pByGenre._2.count(p => p._2) / pByGenre._2.count(p=> (p._2 || !p._2)))
      })
    accuracyByGenre.coalesce(1).saveAsTextFile("/tmp/genreClassification/accuracyByGenre")

    // run execution
    sc.stop()
  }

  /* val testSetFitted : RDD[LabeledPoint] =
     testSet
       .map(ms => new LabeledPoint(genreToDouble(ms.genre), Vectors.dense(ms.synopsis.split("\\W+").map(_.toDouble))))
//    val testSetFitted : RDD[(Double, Vector[String])] = testSet.map(ms => (genreToDouble(ms.genre), ms.synopsis.split("\\W+").toVector))
   val model : NaiveBayesModel =  Classifier.naiveBayesTrainer(sc, trainingSetFitted)
   var prediction : RDD[(Double, Double)] = Classifier.naiveBayesPredicter(sc, testSetFitted, model)
   // run execution
   sc.stop()
 }*/

  /*def genreToDouble(genre : String) : Double = {
    val num = genre match {
      case "Action" => 1
      case "Adventure" => 2
      case "Adult" => 3
      case "Animation" => 4
      case "Comedy" => 5
      case "Crime" => 6
      case "Documentary" => 7
      case "Drama" => 8
      case "Fantasy" => 9
      case "Family" => 10
      case "Film-Noir" => 11
      case "Horror" => 12
      case "Musical" => 13
      case "Mystery" => 14
      case "Romance" => 15
      case "Sci-Fi" => 16
      case "Short" => 17
      case "Thriller" => 18
      case "War" => 19
      case "Western" => 20
    }
    num
  }*/
}
