package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

/**
 * Created by oliver on 13.06.15.
 */
object PreProcessing {

  // line delimiter for plot-summaries list
  val synopsis_line_delim = "-------------------------------------------------------------------------------\n"

  // patterns to extract necessary information from files
  val genre_pattern =
    "^(?![\\s\"])(.+)\\s+\\((\\d{4})\\)\\s+([^\\(\\)][A-Za-z\\-]+)$".r // 1:title, 2:year, 3:genre
  val synopsis_movie_pattern =
    "^MV:\\s+([^\"]+)\\s+\\((\\d{4})\\)\\s+(?!\\(TV\\)|\\(mini\\)|\\(VG\\)|\\(V\\))".r // 1.title, 2:year
  val synopsis_text_pattern =
    "PL:\\s*(.+)".r // 1.one line of the synopsis text

  // fraction of movies that will be in the training set
  val TRAINING_FRACTION = 0.7f


  // create the training and test set
  def preProcess(genrePath: String,
                 synopsisPath: String,
                 sc: SparkContext)
  : (RDD[MovieSynopsis], RDD[MovieSynopsis]) =
  {
    // set up custom line delimiter for synopsis
    val inputConf = new Configuration()
    inputConf.set("textinputformat.record.delimiter", synopsis_line_delim)

    // read files and transform to appropriate RDDs

    // read in files
    val movieSet = extractMovieInfo(genrePath, sc)//, "iso-8859-1"))
    val synopsisSet = extractSynopsisInfo(synopsisPath, sc, inputConf)

    // join RDDs in order to keep only movies that have a synopsis
    val movieSynopsis: RDD[MovieSynopsis] = joinSets(movieSet, synopsisSet)

    //TODO Problem with this: RDD is an abstraction for data that sits on different computers
    //TODO UDFs (user defined functions) like map, reduce, group by get shipped to each computer and can
    //TODO calculate on the partition that is on that computer independently
    //TODO You can't pass arguments that you define in this code to the UDFs
    //TODO and change that argument in the UDF and expect it to be changed overall
    //TODO E.g. you pass trainingSet: Seq[MovieSnopsis] to the foreach loop
    //TODO when the code is shipped to each computer in the spark cluster, it ships the current value of
    //TODO trainingSet which is null(which is also false, you need to initialize it with an empty sequence)
    //TODO then each computer does sth with each empty sequence of trainingSet
    //TODO but it doesn't change the global state of the variable trainingSet you defined below
    val movieSets : RDD[(String, Seq[MovieSynopsis], Seq[MovieSynopsis])]=
      movieSynopsis
        .groupBy((ms : MovieSynopsis) => ms.genre)
        .map(msByGenre => {
        val size = msByGenre._2.size
        val trainingSet : Seq[MovieSynopsis] = msByGenre._2.toSeq.take((size * TRAINING_FRACTION).toInt)
        val testSet : Seq[MovieSynopsis] = msByGenre._2.toSeq.takeRight((size * (1 - TRAINING_FRACTION)).toInt)

        (msByGenre._1, trainingSet, testSet)
      })

    // create training set by keeping TRAINING_FRACTION of movies for each genre
    var trainingSet : Seq[MovieSynopsis]= null
    movieSets.foreach(genreSets => {
      trainingSet.++(genreSets._2)
    })

    // create test set by keeping 1-TRAINING_FRACTION of movies for each genre
    var testSet : Seq[MovieSynopsis]= null
    movieSets.foreach(genreSets => {
      testSet.++(genreSets._3)
    })

    // return (trainingSet, testSet)
    (sc.parallelize(trainingSet), sc.parallelize(testSet) )
  }

  def joinSets(movieSet: RDD[Movie], synopsisSet: RDD[Synopsis]): RDD[MovieSynopsis] = {

    movieSet
      .map(m => ((m.title, m.year), m))
      .join(
        synopsisSet.map(s => ((s.title, s.year), s))
      )
      .map(ms => {
      val title = ms._1._1
      val year = ms._1._2
      MovieSynopsis(title, year, ms._2._1.genre, ms._2._2.synopsis)
    })
  }

  def extractMovieInfo(path: String, sc: SparkContext): RDD[Movie] = {
    sc.textFile(path)
      .flatMap(line => genre_pattern.unapplySeq(line) match {
      case None => Seq.empty[Movie]
      case Some(m) => Seq(new Movie(m(0).toLowerCase.trim, m(1).toInt, m(2).toLowerCase.trim))
    })
  }

  def extractSynopsisInfo(path: String, sc: SparkContext, cfg: Configuration): RDD[Synopsis] = {
    sc.newAPIHadoopFile(
      path,
      classOf[TextInputFormat],
      classOf[LongWritable],
      classOf[Text],
      cfg
    )// ,"iso-8859-1")
    .map(t => t._2.toString)
    .flatMap(line => lineToSynopsis(line))
  }

  def lineToSynopsis(line: String): Seq[Synopsis] = {

    // extract the movie title and year of the synopsis
    val titleYear = synopsis_movie_pattern.findFirstMatchIn(line) match {
      case None => return Seq.empty[Synopsis] // no or invalid movie title -> empty synopsis
      case Some(m) => (m.group(1), m.group(2)) // return tuple(title, year)
    }

    // extract text of the synopsis
    var synopsisText = ""
    synopsis_text_pattern
      .findAllMatchIn(line)
      .foreach(mtch => synopsisText += " " + mtch.group(1))

    Seq(new Synopsis(titleYear._1.toLowerCase.trim, titleYear._2.toInt, synopsisText.toLowerCase.trim))
  }
}
