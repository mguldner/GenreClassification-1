package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.{SparkConf, SparkContext}

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
      .set("spark.driver.allowMultipleContexts", "true")
    val sc = new SparkContext(conf)


    val synopsi = sc.
      newAPIHadoopFile(
        synopsisPath,
        classOf[TextInputFormat],
        classOf[LongWritable],
        classOf[Text],
        inConf
    )

    synopsi
      .map(t => t._2.toString)
      .coalesce(1)
      .saveAsTextFile("test.txt")

    // create two datasets of MovieSynopsis (as (trainingSet, testSet))
    /*val movieSet = PreProcessing.extractMovieInfo(sc.textFile(genrePath))//, "iso-8859-1"))
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
      .saveAsTextFile("file:///tmp/genreclass/join.list")*/

    // run execution
    sc.stop()
  }
}
