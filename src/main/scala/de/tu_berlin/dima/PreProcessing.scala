package de.tu_berlin.dima

import chalk.text.tokenize.SimpleEnglishTokenizer
import chalk.text.tokenize.SimpleEnglishTokenizer.V1
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector

import scala.util.Random

object SVMParam extends Enumeration {
  type SVMParam = Value
  val REG, NUM_IT, STEP_SIZE = Value
}

object TfidfType extends Enumeration {
  type TfidfType = Value
  val SPARK_TOT, SPARK_TF, TUPLE = Value
}

object PreProcessing {

  // field delimiter after processing synopses and writing them out
  val SYN_DELIM = "ID"
  val TAB = "TD"

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
    val trainTest = splitTrainingTest(movieSynopsis)

    (trainTest._1, trainTest._2)
  }

  def preProcess(movieSynopses: RDD[MovieSynopsis])
  : (RDD[((String, Int, Seq[String]), Vector)], RDD[((String, Int, Seq[String]), Vector)]) =
  {

    val splits = movieSynopses.randomSplit(Array(TRAINING_FRACTION, 1-TRAINING_FRACTION), 11L)
    val trIn = splits(0)//PreProcessing.transformToMovieSynopsis(sc.textFile(train))
    val t = splits(1)//PreProcessing.transformToMovieSynopsis(sc.textFile(test))
    val tsIn = getAllGenres(trIn, t)

    // split into training and test
    //val trTs = PreProcessing.splitTrainingTest(movieSynopses)
    val tr = trIn/*trTs._1*/.map(ms => ((ms.title,ms.year,Seq(ms.genre)), ms.synopsis))
    val ts = tsIn/*trTs._2*/.map(ms => ((ms._1._1,ms._1._2,ms._1._3), ms._2))//.map(ms => ((ms.title,ms.year,ms.genre), ms.synopsis))

    // apply tfidf transformation
    val tfidfTransformer = new TFIDF()
    val tfidfTr = tfidfTransformer.tfidfVectorParadigm(tr)
    val tfidfTs = tfidfTransformer.tfidfVectorParadigm(ts)

    // normalize vectors (to account for different lengths of movie synopses)
    val norm = new Normalizer()
    val normTfidfTr = tfidfTr//.map(p => (p._1, norm.transform(p._2)))
    val normTfidfTs = tfidfTs//.map(p => (p._1, norm.transform(p._2)))

    (normTfidfTr, normTfidfTs)
  }

  def filterGenres(movies: RDD[MovieSynopsis], genresToFilter: Seq[String]): RDD[MovieSynopsis] = {

    if(genresToFilter.size == 1 && genresToFilter.head.equals("*")) {
      println("get all genres")
      movies
    } else {
      movies
        .filter(m => genresToFilter.contains(m.genre))
    }
  }

  def getAllGenres(train: RDD[MovieSynopsis],
                   test: RDD[MovieSynopsis])
  : RDD[((String, Int, Seq[String]), Seq[String])] =
  {
    train.map(ms => (ms, false))
      .union(test.map(ms => (ms, true)))
      .map(ms => ((ms._1.title, ms._1.year), ms))
      .groupByKey()
      .flatMap(ms => {
        val title = ms._1._1
        val year = ms._1._2
        val it = ms._2.toList
        val syn = it.head._1.synopsis
        var genres = Seq[String]()
        var isFromSubset = false
        it.foreach(s => {
          // at least one element has to come from the test set
          if(s._2) isFromSubset = true
          genres +:= s._1.genre
        })
        if(isFromSubset)
          Seq(((title, year, genres), syn))
        else
          Seq.empty[((String, Int, Seq[String]), Seq[String])]
      })
  }

  def transformFromMovieSynopsis(movies: RDD[MovieSynopsis]): RDD[String] = {
    movies.map(ms => {
      var syn = ms.title+SYN_DELIM+ms.year+SYN_DELIM+ms.genre+TAB
      var first = true
      ms.synopsis.foreach(word => {
        if(first) {
          syn += word
          first = false
        } else {
          syn += SYN_DELIM+word
        }
      })
      syn
    })
  }

  def transformToMovieSynopsis(movies: RDD[String]): RDD[MovieSynopsis] = {
    movies.map(ms => {
      val spl = ms.split(TAB)
      require(spl.size == 2, "split has to be [id, synopsis]: " + spl.size + ";" + spl.toString)
      val id = spl(0).split(SYN_DELIM)
      require(id.size == 3, "id is title,year,genre: " + id.toString)
      val syn = spl(1).split(SYN_DELIM).toSeq

      MovieSynopsis(id(0), id(1).toInt, id(2), syn)
    })
  }

  def splitTrainingTest(movies: RDD[MovieSynopsis]): (RDD[MovieSynopsis], RDD[MovieSynopsis]) = {
    // put for each genre a fraction into the training and a 1-fraction into the test set
    val movieSets = movies
      .map(ms => {
        val p = Random.nextDouble()
        (ms, p <= TRAINING_FRACTION) // if true --> part of training set, else part of test set
      })
      /*.groupBy((ms : MovieSynopsis) => ms.genre)
      .flatMap(msByGenre => {
      msByGenre._2 // get synopses for genre
        .map(ms => {
        val p = Random.nextDouble()
        (ms, p <= TRAINING_FRACTION) // if true --> part of training set, else part of test set
      })
    })*/

    val trainingSet = movieSets
      .flatMap(mb => mb._2 match {
        case true => Seq(mb._1)
        case false => Seq.empty[MovieSynopsis]
      })

    val testSet = movieSets
      .flatMap(mb => mb._2 match {
        case true => Seq.empty[MovieSynopsis]
        case false => Seq(mb._1)
      })

    /*val spl = movies
      .filter(m => m.genre.equals(PreProcessing.doubleToGenre(1)))
      .randomSplit(Array(TRAINING_FRACTION, 1-TRAINING_FRACTION), 11L)

    var tr = spl(0)
    var ts = spl(1)

    for(i <- 2 to 24) {
      val splits = movies
        .filter(m => m.genre.equals(PreProcessing.doubleToGenre(i)))
        .randomSplit(Array(TRAINING_FRACTION, 1-TRAINING_FRACTION), 11L)

      tr = tr.union(splits(0))
      ts = ts.union(splits(1))
    }

    (tr, ts)*/
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

  def moviesPerGenre(data: RDD[String]): Map[String, Int] = {
    data
      .map(g => (g, 1))
      .reduceByKey((x, y) => x+y)
      .collectAsMap()
      .toMap
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
      case "reality-tv" => 25
      case "news" => 26
      case "talk-show" => 27
      case "game-show" => 28

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
      case  25 => "reality-tv"
      case  26 => "news"
      case  27 => "talk-show"
      case  28 => "game-show"
      case other => throw new RuntimeException("Wrong genre: " + other)
    }
    str
  }

  // process a movie synopsis represented by a string to a synopsis represented by its words
  def processSynopsis(syn: String): Seq[String] = {
    SimpleEnglishTokenizer
      .V1()
      .apply(syn)
      .toList
  }
}
