package de.tu_berlin.dima

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import scala.reflect.ClassTag
import scala.util.hashing.MurmurHash3


case class TfidfScores[I: ClassTag](docId: I, hashWord: Int, tf: Int, idf: Double, tfidf: Double)

/**
 * Transform a RDD of documents, each document represented as a sequence of its' words,
 * to an RDD of documents, each document represented as a sequence of the its' words' TFIDF scores
 */
class TFIDF extends Serializable {

  def tfidfVectorParadigm[D <: Iterable[String], I <: Product](documents: RDD[(I, D)],
                                                               useIdfVec: Boolean = true,
                                                               numFts: Int = Math.pow(2,10).toInt)
  : RDD[(I, Vector)] =
  {

    val tf = new StringTF(numFts)
      .transform(documents)

    tf.cache()
    val idf = new StringIDF(useIdfVec)
      .fit(tf.map(d => d._2)) // only features necessary for IDF calculation

    val tfidf: RDD[(I, Vector)] = idf.transform(tf)

    tfidf
  }

  def tfidfTupleParadigm[D <: Iterable[String], I : ClassTag](documents: RDD[(I, D)])
  : RDD[(I, Vector)] =
  {
    //TODO due to lazy evaluation this doesn't work as it is
    //TODO mybe calculate df first or just count the number of documents in one map recude job and broadcast that
    //TODO as below
    //val numDocs = documents.context.accumulator(0, "numDocs")

    // need to create broadcast variable that encodes the number of docs
    val numDocs = documents
      .map(t => 1)
      .reduce((x,y) => x+y)
    val numDocsBc = documents.context.broadcast(numDocs)

    // compute Term frequencies
    val tfs = documents.flatMap(doc => {
        //numDocs += 1
        toTfidfScore(doc._1, doc._2)
      })

    // group by word and compute tfidf scores
    val tfidfs = tfs
          .map(tf => (tf.hashWord, tf))
          .groupByKey()
          .flatMap(wordGroup => {
            val hashwrd = wordGroup._1
            val tfidfscrs = wordGroup._2.toSeq
            val df = tfidfscrs.size
            tfidfscrs.map(scrs => {

              val ndcs = numDocsBc.value
              val idf = Math.log((ndcs + 1d) / (df + 1d))

              TfidfScores(
                docId = scrs.docId,
                hashWord = hashwrd,
                tf = scrs.tf,
                idf = idf,
                tfidf = scrs.tf * idf
              )
            })
          })

    // Spark algorithms take vectors as input

    // convert to document vectors
    tfidfs
      // group by document id
      .map(tf => (tf.docId, tf))
      .groupByKey()
      // transform document group to sparse vector
      .map(docGroup => {
        val docId = docGroup._1
        val scrs = docGroup
          ._2.map(tfidf => (tfidf.hashWord, tfidf.tfidf))
          .toSeq

        (docId, Vectors.sparse(Int.MaxValue, scrs))
      })
  }

  /**
   * Transforms the input document into a sequence of tfidfscores
   * (docId, hashValue)
   */
  private def toTfidfScore[I: ClassTag](docId: I, document: Iterable[String])
  : Iterable[TfidfScores[I]] =
  {
    val termFrequencies = scala.collection.mutable.HashMap.empty[Int, Int]
    document.foreach { term =>
      val i = Math.abs(MurmurHash3.stringHash(term))
      // increase term frequency count
      termFrequencies.put(i, termFrequencies.getOrElse(i, 0) + 1)
    }

    // create tfidf scores
    termFrequencies.map(tf => TfidfScores(docId, tf._1, tf._2, 1, -1))
  }
}
