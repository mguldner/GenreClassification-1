package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.spark.{SparkContext, SparkConf}


object TfidfAnalysis {

  def main(args: Array[String]) {

    // set up global parameters
    if (args.length != 3)
      throw new RuntimeException(
        "usage: [path-to-movies.list] [tfidf-type] [num features]"
      )
    val moviesPath = args(0)
    val tfidfType = TfidfType(args(1).toInt)
    val numFeatures = args(2).toInt

    // set up exectution
    val conf = new SparkConf()
      .setAppName("Tfidf Analysis")
    val sc = new SparkContext(conf)


    // create movie synopsis dataset
    // read in files
    val movieSet = PreProcessing
      .transformToMovieSynopsis(sc.textFile(moviesPath))
      .map(m => ((m.title, m.year, m.genre), m.synopsis))

    val numDocsBefore = movieSet.count()
    println("Number of Docs before transformation: " + numDocsBefore)

    // run tfidf transformer on dataset
    val tfidfTransf = new TFIDF()

    val results = tfidfType match {
      case TfidfType.SPARK_TOT => tfidfTransf.tfidfVectorParadigm(movieSet, true, numFeatures)
      case TfidfType.SPARK_TF => tfidfTransf.tfidfVectorParadigm(movieSet, false, numFeatures)
      case TfidfType.TUPLE => tfidfTransf.tfidfTupleParadigm(movieSet)
    }

    val numDocs = results.count()
    println("Number of Docs after transformation: " + numDocs)

    sc.stop()
  }
}
