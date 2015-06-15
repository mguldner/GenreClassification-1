package de.tu_berlin.dima


import org.apache.spark.{SparkContext, SparkConf}
import org.junit.{After, Before, Test}

import scala.reflect.io.File

/**
 * Created by oliver on 15.06.15.
 */
class PreProcessingTest {

  val conf = new SparkConf().setAppName("PreProcessingTest").setMaster("local[4]")
  val sc = new SparkContext(conf)
  val genrePath = "genres.list"
  val synopsisPath = "plot.list"
  val tempDirProp = "java.io.tmpDir"


  @Before
  def setUp(): Unit = {

    // write movies to genre
    File(genrePath).writeAll(
      // good movies
      "La seule sortie (2010)\t\t\t\t\tAnimation\nLa seule sortie (2010)\t\t\t\t\tHorror\nLa seule sortie (2010)\t\t\t\t\tShort\n" +
        "#Murder (2015)\tComedy\nThe Hunger Games: Fire of Darkness (2016)\tThriller\n" +
        "The Hunger Games: Mockingjay - Part 1 (2014)\t\tAdventure\n" +

        // tv shows or taged titles
        "\"Phil of the Future\" (2004)\t\t\t\tAdventure\n" +
        "La sidra vive en Nava (2004) (TV)\t\t\tDocumentary\nLa siciliana (2000) (V)\t\t\t\t\tAdult" +
        "#IDARB (It Draws a Red Box) (2015) (VG)\t\t\tAction\n"
    )

    // write movies to synopsis
    File(synopsisPath).writeAll(
      // good plot summaries
      "MV: American Sniper (2014)\n\nPL: Chris Kyle was nothing more than a Texan man who wanted to become a cowboy,\n" +
        "PL: but in his thirties he found out that maybe his life needed something\nBY: Evandro Martirano\n" +
        "PL: Navy SEAL sniper Chris Kyle pinpoint accuracy saves countless lives on the\n" +
        "PL: battlefield and turns him into a legend. Back home to his wife and kids\n" +
        "BY: Claudio Carvalho, Rio de Janeiro, Brazil\n\n-------------------------------------------------------------------------------\n" +

        // tv shows, flagged titles
        "MV: American Bounty Hunter (1996) (TV)\n\n" +
        "PL: This action-packed one hour special is an explosive response to America's\n" +
        "PL: hunger for reality-based TV programming. In the wake of such successful TV\n" +
        "BY: Rod Mitchell <mediarod@aol.com>\n\n-------------------------------------------------------------------------------\n" +
        "MV: \"#7DaysLater\" (2013)\n\nPL: #7dayslater is an interactive comedy series featuring an ensemble cast of\n" +
        "BY: Sum Whan\n\n-------------------------------------------------------------------------------"
    )
  }

  @After
  def cleanUp(): Unit = {
    File(genrePath).deleteRecursively()
    File(synopsisPath).deleteRecursively()

    assert(!File(synopsisPath).exists)
    assert(!File(genrePath).exists)
  }

  @Test
  def testParsing(): Unit = {
    val parsedGenres = PreProcessing.extractMovieInfo(env.readTextFile(genrePath, "iso-8859-1")).collect()
    val parsedSynopses = PreProcessing.extractSynopsisInfo(
      sc.readFile(new CustomInputFormat("iso-8859-1", PreProcessing.synopsis_line_delim), synopsisPath)
    ).collect()

    // assert correct genres
    assert(parsedGenres.contains(Movie("la seule sortie", 2010, "animation")))
    assert(parsedGenres.contains(Movie("la seule sortie", 2010, "horror")))
    assert(parsedGenres.contains(Movie("la seule sortie", 2010, "short")))
    assert(parsedGenres.contains(Movie("#murder", 2015, "comedy")))
    assert(parsedGenres.contains(Movie("the hunger games: fire of darkness", 2016, "thriller")))
    assert(parsedGenres.contains(Movie("the hunger games: mockingjay - part 1", 2014, "adventure")))
    assert(parsedGenres.length == 6)

    // assert correct synopses
    val text = "chris kyle was nothing more than a texan man who wanted to become a cowboy, " +
      "but in his thirties he found out that maybe his life needed something " +
      "navy seal sniper chris kyle pinpoint accuracy saves countless lives on the " +
      "battlefield and turns him into a legend. back home to his wife and kids"

    assert(parsedSynopses.contains(Synopsis("american sniper", 2014, text)))
    assert(parsedSynopses.length == 1)
  }

  @Test
  def testJoin(): Unit = {
    val movieSet = sc.parallelize(
      List(
        // different genre, rest is the same
        Movie("forrest gump", 1994, "drama"),
        Movie("forrest gump", 1994, "comedy"),

        // different years
        Movie("harry potter", 2012, "adventure"),
        Movie("harry potter", 2001, "fantasy"),

        // not contained
        Movie("moonstruck", 1984, "romance")
      )
    )

    val synopsisSet = sc.parallelize(
      List(
        Synopsis("forrest gump", 1994, "great movie"),
        Synopsis("harry potter", 2012, "good movie"),
        Synopsis("harry potter", 2001, "also a good movie")
      )
    )

    val joinSet = PreProcessing.joinSets(movieSet, synopsisSet).collect()

    assert(joinSet.contains(MovieSynopsis("forrest gump", 1994, "drama", "great movie")))
    assert(joinSet.contains(MovieSynopsis("forrest gump", 1994, "comedy", "great movie")))
    assert(joinSet.contains(MovieSynopsis("harry potter", 2001, "fantasy", "also a good movie")))
    assert(joinSet.contains(MovieSynopsis("harry potter", 2012, "adventure", "good movie")))
    assert(joinSet.length == 4)
  }
}
