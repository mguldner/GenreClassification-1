/*
  Copyright 2015 Oliver Breit

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
 */

package de.tu_berlin.dima

import java.util.stream.Collector

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.immutable.HashMap


/**
 * Very important if document corpus contains a lot of words
 * and we choose a large dimensions number for the TF
 * IDF vector is broadcast!!
 *
 * Problem: Number of words in the english language: 1,025,109 (google)
 * Size of Int (assuming 32bit) = 4byte
 * --> if our document corpus contains all words in the english language
 * we get a IDF vector of approx. (1,025,109*4)byte ~ 4MB
 * --> good to broadcast
 *
 * But! our feature dimensions is 2^31 because MurmurHash gives back an integer
 * --> this IDF implementation creates a a dense! array with size equal to the number of features
 * --> this would be 17GB for 2^31
 * --> QuickFix: modulo the size of murmur hash to the desired number of features
 * --> for {{{2^20}}} we get a 8MB array which is ok
 */
class StringIDF(val minDocFreq: Int, val useIdfVector: Boolean) {

  def this() = this(0, true)

  def this(useIdfVec: Boolean) = this(0, useIdfVec)

  /**
   * Computes the inverse document frequency.
   * @param tfVectors an RDD of term frequency vectors
   */
  def fit(tfVectors: RDD[Vector]): StringIDFModel = {

    // use IDF as a vector --> Spark version
    if(useIdfVector) {
      val idf = tfVectors.treeAggregate(new StringIDF.DocumentFrequencyAggregator(
        minDocFreq = minDocFreq))(
          seqOp = (df, v) => df.add(v),
          combOp = (df1, df2) => df1.merge(df2)
        ).idf()
      new StringIDFModel(idf)
    }
    // use IDF as map --> own version
    else
    {
      // create accumulator for counting number of documents
      val numDocs = tfVectors
        .context
        .accumulator(0l, "numDocs")

      // count document frequency of words and total number of documents via accumulator variable
      val wordsCount = tfVectors
        .flatMap(v => v match {
          case SparseVector(s, indices, vals) => {
            // accumulate the number of documents
            numDocs += 1
            indices.map(hv => (hv, 1))
          }
          case other => throw new RuntimeException("tfVectors have to be of type sparse vector: " + other)
        }) // indices represent hashed words
        .reduceByKey((x,y) => {

          x+y
        }) // count in how many documents a word appears

      // collect IDF vector
      // compute IDF scores for each word
      val idfMap = wordsCount
        .collectAsMap()
        .flatMap(df => df._2 >= minDocFreq match {
          case true => Seq((df._1, idf(numDocs.value, df._2)))
          case false => Seq.empty[(Int, Double)]
        })
        .toMap

      new StringIDFModel(idfMap)
    }
  }

  // compute idf score
  private def idf(numDocs: Long, df: Long): Double = {
    Math.log((numDocs + 1d) / (df + 1d))
  }
}

private object StringIDF {

  /** Document frequency aggregator. */
  class DocumentFrequencyAggregator(val minDocFreq: Int) extends Serializable {

    /** number of documents */
    private var m = 0L
    /** document frequency vector */
    private var df: BDV[Long] = _

    def this() = this(0)

    /** Adds a new document. */
    def add(doc: Vector): this.type = {
      if (isEmpty) {
        // doc.size == number of features --> for large number of features this df vector can get unnecessarily large
        df = BDV.zeros(doc.size)
      }
      doc match {
        case SparseVector(size, indices, values) =>
          val nnz = indices.length
          var k = 0
          while (k < nnz) {
            if (values(k) > 0) {
              df(indices(k)) += 1L
            }
            k += 1
          }
        case DenseVector(values) =>
          val n = values.length
          var j = 0
          while (j < n) {
            if (values(j) > 0.0) {
              df(j) += 1L
            }
            j += 1
          }
        case other =>
          throw new UnsupportedOperationException(
            s"Only sparse and dense vectors are supported but got ${other.getClass}.")
      }
      m += 1L
      this
    }

    /** Merges another. */
    def merge(other: DocumentFrequencyAggregator): this.type = {
      if (!other.isEmpty) {
        m += other.m
        if (df == null) {
          df = other.df.copy
        } else {
          df += other.df
        }
      }
      this
    }

    private def isEmpty: Boolean = m == 0L

    /** Returns the current IDF vector. */
    def idf(): Vector = {
      if (isEmpty) {
        throw new IllegalStateException("Haven't seen any document yet.")
      }
      val n = df.length
      val inv = new Array[Double](n)
      var j = 0
      while (j < n) {
        /*
         * If the term is not present in the minimum
         * number of documents, set IDF to 0. This
         * will cause multiplication in IDFModel to
         * set TF-IDF to 0.
         *
         * Since arrays are initialized to 0 by default,
         * we just omit changing those entries.
         */
        if (df(j) >= minDocFreq) {
          inv(j) = math.log((m + 1.0) / (df(j) + 1.0))
        }
        j += 1
      }
      Vectors.dense(inv)
    }
  }
}

class StringIDFModel (val idf: Vector) extends Serializable {

  var idfMap: IntDoubleMap = IntDoubleMap(Map[Int, Double]())

  /**
   * Constructing IDF model from hashmap
   * @param idf IDF vector as hashmap
   *            Key = hashed word
   *            Value = IDF score
   */
  def this(idf: Map[Int, Double]) = {
    this(Vectors.dense(Array.empty[Double]))
    idfMap = IntDoubleMap(idf)
  }

  /**
   * Transforms term frequency (TF) vectors to TF-IDF vectors.
   *
   * If `minDocFreq` was set for the IDF calculation,
   * the terms which occur in fewer than `minDocFreq`
   * documents will have an entry of 0.
   *
   * Use either IDF as a vector or as a map
   *
   * @param dataset an RDD of term frequency vectors
   * @return an RDD of TF-IDF vectors
   */
  def transform[I <: Product](dataset: RDD[(I, Vector)]): RDD[(I, Vector)] = {

    if(idf.size == 0 && idfMap.map.size == 0) {
      throw new RuntimeException("IDF vector and IDF map are not set")
    }

    // use IDF vector
    if(idf.size != 0) {
      println("Use IDF Vector")
      val bcIdf = dataset.context.broadcast(idf)
      return dataset.mapPartitions(iter => iter.map(v => (v._1, StringIDFModel.transform(bcIdf.value, v._2))))
    }

    // use IDF map
    println("Use IDF Map")
    val bcIdf = dataset.context.broadcast(idfMap)
    dataset.mapPartitions(iter => iter.map(v => (v._1, StringIDFModel.transform(bcIdf.value, v._2))))
  }
}

// wrapper around Map[Int, Double] --> for save pattern matching
case class IntDoubleMap(map: Map[Int, Double])

private object StringIDFModel {

  /**
   * get the idf score based on whether IDF is a vector or map
   * @param idf IDF collection
   * @param indx hashed word
   * @return IDF score for idf(indx) or 0 if it doesn't exist
   */
  def idfScore(idf: Any, indx: Int): Double = {

    idf match {
      case m: IntDoubleMap => m.map.getOrElse(indx, 0.0)
      case v: org.apache.spark.mllib.linalg.Vector => v(indx)
      case other => {
        val c = other.getClass//.toString
        throw new RuntimeException("idf is of type " + c)
      }
    }
  }

  /**
   * Transforms a term frequency (TF) vector to a TF-IDF vector with an
   * IDF collection
   *
   * @param idf an IDF collection
   * @param tf a term frequency vector
   * @return a TF-IDF vector
   */
  def transform(idf: Any, tf: Vector): Vector = {
    val n = tf.size
    tf match {
      case SparseVector(size, indices, values) =>
        val nnz = indices.length
        val newValues = new Array[Double](nnz)
        var k = 0
        while (k < nnz) {
          // index the IDF vector with the hash value of the word which, in turn, is the index for the TF vector
          newValues(k) = values(k) * idfScore(idf, indices(k))
          k += 1
        }
        Vectors.sparse(n, indices, newValues)
      case DenseVector(values) =>
        val newValues = new Array[Double](n)
        var j = 0
        while (j < n) {
          newValues(j) = values(j) * idfScore(idf, j)
          j += 1
        }
        Vectors.dense(newValues)
      case other =>
        throw new UnsupportedOperationException(
          s"Only sparse and dense vectors are supported but got ${other.getClass}.")
    }
  }
}

