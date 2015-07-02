package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vector

/**
 * Created by oliver on 13.06.15.
 */
object PreProcessingMain {
  //TODO same as in other class with line delim
  def main(args: Array[String]) {

    // input argument flags
    val genresPathFlag = "genres"
    val synopsisPathFlag = "synopses"

    // set up global parameters
    if(args.length != 2) throw new RuntimeException("usage: [path-to-genres.list] [path-to-synopses.list]")
    val genrePath = args(0)
    val synopsisPath = args(1)

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
      .saveAsTextFile("file:///home/oliver/Documents/datasets/genre/out/trainingSet")
    test
      .map(m => (m.title, m.year, m.genre))
      .coalesce(1)
      .saveAsTextFile("file:///home/oliver/Documents/datasets/genre/out/testSet")

    /*val sets : (RDD[MovieSynopsis], RDD[MovieSynopsis]) = PreProcessing.preProcess(genrePath, synopsisPath, sc)
    val trainingSet : RDD[MovieSynopsis] = sets._1
    val testSet : RDD[MovieSynopsis] = sets._2
    val useIdfVec : Boolean = true

    val trainingSetChanged : RDD[(String, Iterable[String])] =
      trainingSet.map(ms => (ms.genre, ms.synopsis.split("\\W+").toIterable))

    var trainingSetFitted : RDD[LabeledPoint] =
      TFIDF
        .tfidfVectorParadigm(trainingSetChanged, useIdfVec)
        .map(v => new LabeledPoint(genreToDouble(v._1), v._2))

    var model : NaiveBayesModel =  Classifier.naiveBayesTrainer(sc, trainingSetFitted)
    var prediction : RDD[(Double, Double)] = Classifier.naiveBayesPredicter(sc, testSet, model)*/
    // run execution
    sc.stop()
  }
}
