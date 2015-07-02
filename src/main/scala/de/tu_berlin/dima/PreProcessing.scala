package de.tu_berlin.dima

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD

import scala.util.Random

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
  val TRAINING_FRACTION = 0.8f

  // transform a RDD of MovieSynopis to RDD of Labeled Point
  def tfidfTransformation(movies: RDD[MovieSynopsis]): RDD[LabeledPoint] = {

    // transform documents to necessary tfidf input
    val msForTfidf = movies
      .map(m => ((m.title, m.year, m.genre), m.synopsis))

    // apply tfidf transformation
    val tfidf = new TFIDF()
    tfidf
      .tfidfVectorParadigm(msForTfidf)
      .map(doc => LabeledPoint(genreToDouble(doc._1._3), doc._2))
  }

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

    // split into training and test set
    val movieSets = movieSynopsis
        .groupBy((ms : MovieSynopsis) => ms.genre)
        .flatMap(msByGenre => {
          msByGenre._2 // get synopses for genre
            .map(ms => {
              val p = Random.nextDouble()
              (ms, p <= TRAINING_FRACTION) // if true --> part of training set, else part of test set
            })
        })

    val trainingSet = movieSets
      .filter(mb => mb._2) // keep movie if part of training set
      .map(mb => mb._1)

    val testSet = movieSets
      .filter(mb => !mb._2) // keep movie if part of test set
      .map(mb => mb._1)

    (trainingSet, testSet)
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
        MovieSynopsis(title, year, ms._2._1.genre, processSynopsis(ms._2._2.synopsis))
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

  def genreToDouble(genre : String) : Double = {
    val num = genre match {
      case "short" => 1
      case "drama" => 2
      case "comedy" => 3
      case "documentary" => 4
      case "adult" => 5
      case "action" => 6
      case "romance" => 7
      case "thriller" => 8
      case "animation" => 9
      case "family" => 10
      case "horror" => 11
      case "music" => 12
      case "crime" => 13
      case "adventure" => 14
      case "fantasy" => 15
      case "sci-fi" => 16
      case "mystery" => 17
      case "biography" => 18
      case "history" => 19
      case "sport" => 20
      case "musical" => 21
      case "war" => 22
      case "western" => 23
      case "film-noir" => 24
      /*case "reality-tv" => 25
      case "news" => 26
      case "talk-show" => 27
      case "game-show" => 28*/

      case other => throw new RuntimeException("Wrong genre: " + other)
    }
    num
  }

  def doubleToGenre(dgenre: Double): String = {
    val str = dgenre match {
      case  1 => "short"
      case  2 => "drama"
      case  3 => "comedy"
      case  4 => "documentary"
      case  5 => "adult"
      case  6 => "action"
      case  7 => "romance"
      case  8 => "thriller"
      case  9 => "animation"
      case  10 => "family"
      case  11 => "horror"
      case  12 => "music"
      case  13 => "crime"
      case  14 => "adventure"
      case  15 => "fantasy"
      case  16 => "sci-fi"
      case  17 => "mystery"
      case  18 => "biography"
      case  19 => "history"
      case  20 => "sport"
      case  21 => "musical"
      case  22 => "war"
      case  23 => "western"
      case  24 => "film-noir"
      /*case "reality-tv" => 25
      case "news" => 26
      case "talk-show" => 27
      case "game-show" => 28*/
      case other => throw new RuntimeException("Wrong genre: " + other)
    }
    str
  }

  // process a movie synopsis represented by a string to a synopsis represented by its words
  def processSynopsis(syn: String): Seq[String] = {
    syn.split("\\W+")
  }
}
