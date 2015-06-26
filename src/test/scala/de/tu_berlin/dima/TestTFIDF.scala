package de.tu_berlin.dima

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.Test

/**
 * Created by oliver on 25.06.15.
 */
class TestTFIDF {

  val conf = new SparkConf().setAppName("TestTFIDF").setMaster("local[4]")
  val sc = new SparkContext(conf)

  @Test
  def testTfidf(): Unit = {
    val docs = sc.parallelize(
      Seq(
        (("forrest gump", 1994, "comedy"), Seq("he","love","and","lost","and","love","and","chocolate")),
        (("the notebook", 2005, "romance"), Seq("love", "girl", "boy", "and", "rain", "love", "love", "excitement")),
        (("jurrasic world", 2015, "action"), Seq("danger", "and", "excitement", "and", "lost", "limbs", "excitement"))
      )
    )

    // test spark
    assert(applyTfidf(docs, Math.round(Math.pow(2,20)).toInt, true))

    // test own impl
    assert(applyTfidf(docs, Int.MaxValue, false))

    // test tuple paradigm
    val tfidf = new TFIDF()
    assert(testCorrectness(tfidf.tfidfTupleParadigm(docs).collect()))
  }

  def applyTfidf(docs: RDD[((String, Int, String), Seq[String])],
                 numFts: Int,
                 useIdfVec: Boolean)
  : Boolean =
  {
    val tfidfDocs = new TFIDF()
      .tfidfVectorParadigm(docs, useIdfVec, numFts)
      .collect()

    testCorrectness(tfidfDocs)
  }

  def testCorrectness(tfidfDocs: Seq[((String, Int, String), org.apache.spark.mllib.linalg.Vector)])
  : Boolean =
  {
    tfidfDocs.foreach(d => {
      println("Movie: " + d._1._1 + "; " + d._2.toString)
    })

    val scores = tfidfDocs.flatMap(tfidf => {
      val vec = tfidf
        ._2.asInstanceOf[SparseVector]

      var flatscrs: Seq[Double] = Seq[Double]()
      vec.values.foreach(v => {
        flatscrs = flatscrs.:+(v)
      })

      flatscrs
    })

    (scores.count(s => BigDecimal(s).setScale(2, BigDecimal.RoundingMode.HALF_EVEN).toDouble == 0.69) == 7) &&
    (scores.count(s => BigDecimal(s).setScale(2, BigDecimal.RoundingMode.HALF_EVEN).toDouble == 0.58) == 2) &&
    (scores.count(s => BigDecimal(s).setScale(2, BigDecimal.RoundingMode.HALF_EVEN).toDouble == 0.29) == 3) &&
    (scores.count(s => BigDecimal(s).setScale(2, BigDecimal.RoundingMode.HALF_EVEN).toDouble == 0.86) == 1) &&
    (scores.count(s => BigDecimal(s).setScale(2, BigDecimal.RoundingMode.HALF_EVEN).toDouble == 0.00) == 3)
  }
}

