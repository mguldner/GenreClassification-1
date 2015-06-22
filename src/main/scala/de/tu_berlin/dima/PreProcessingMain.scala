package de.tu_berlin.dima

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
    val conf = new SparkConf()
      .setAppName("Pre Processing")
      .setMaster("local[4]")
    val sc = new SparkContext()

    // create two datasets of MovieSynopsis (as (trainingSet, testSet))
    val movieSet = PreProcessing.extractMovieInfo(sc.textFile(genrePath))//, "iso-8859-1"))
    movieSet
      .coalesce(1)
      .saveAsTextFile("file:///tmp/genreclass/genres.list")

    val synopsisSet = PreProcessing.extractSynopsisInfo(
      sc.textFile(synopsisPath)//(new CustomInputFormat("iso-8859-1", PreProcessing.synopsis_line_delim), synopsisPath)
    )
    synopsisSet
      .coalesce(1)
      .saveAsTextFile("file:///tmp/genreclass/plot.list")

    PreProcessing
      .joinSets(movieSet, synopsisSet)
      .coalesce(1)
      .saveAsTextFile("file:///tmp/genreclass/join.list")

    // Creation of the two datasets (trainingSet and testSet)
    val sets : (RDD[MovieSynopsis], RDD[MovieSynopsis]) = PreProcessing.preProcess(genrePath, synopsisPath, sc)
    val tfidf : RDD[Vector] = TFIDF.apply(sets._1)

    // run execution
    sc.stop()
  }
}
