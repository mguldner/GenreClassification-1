package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vector

import scala.util.Random


/**
 * Created by oliver on 13.06.15.
 */
object PreProcessingMain {

  def main(args: Array[String]) {

    // input argument flags
    val genresPathFlag = "genres"
    val synopsisPathFlag = "synopses"

    // set up global parameters
    if(args.length != 4) throw new RuntimeException("usage: [path-to-genres.list] [path-to-synopses.list] [path-to-joined-set] [genres]")
    val genrePath = args(0)
    val synopsisPath = args(1)
    val outPath = args(2)
    val genres = args(3).split(",")

    // synopis file needs a custom line delimiter as one synopsis goes over multiple lines
    val inConf = new Configuration()
    inConf.set("textinputformat.record.delimiter", PreProcessing.synopsis_line_delim)

    // set up exectution
    val conf = new SparkConf().setAppName("Pre Processing")
    val sc = new SparkContext(conf)

    // create movie synopsis set
    val movieSet = PreProcessing
      .extractMovieInfo(genrePath, sc)//, "iso-8859-1"))
      .filter(m => genres.contains(m.genre))
    val synopsisSet = PreProcessing.extractSynopsisInfo(synopsisPath, sc, inConf)

    // join RDDs in order to keep only movies that have a synopsis
    val movieSynopses: RDD[MovieSynopsis] = PreProcessing.joinSets(movieSet, synopsisSet)

    // write out movie synopsis set
    PreProcessing
      .transformFromMovieSynopsis(movieSynopses)
      .coalesce(1)
      .saveAsTextFile(outPath)
    /*val ms = PreProcessing
      .transformToMovieSynopsis(sc.textFile(args(0)))
    val splits = PreProcessing.splitTrainingTest(ms)
    val tr = splits._1
    val ts = splits._2

    PreProcessing.transformFromMovieSynopsis(tr).coalesce(1).saveAsTextFile("file:///home/oliver/Documents/datasets/genre/train/rain_dramcom")
    PreProcessing.transformFromMovieSynopsis(ts).coalesce(1).saveAsTextFile("file:///home/oliver/Documents/datasets/genre/test/est_dramcom")*/

    sc.stop()
  }
}
