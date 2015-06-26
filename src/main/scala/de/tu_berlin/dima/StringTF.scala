package de.tu_berlin.dima

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.hashing.MurmurHash3

/**
 * TF based on Spark's implementation
 * This TF works specifically on Strings but allows to keep the document id for each document
 */
class StringTF(val numFeatures: Int) extends Serializable
{

  def this() = this(Math.pow(2, 20).toInt) // default

  /* Calculates 'x' modulo 'mod', takes to consideration sign of x,
   * i.e. if 'x' is negative, than 'x' % 'mod' is negative too
   * so function return (x % mod) + mod in that case.
   */
  private def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  // Handles idiosyncracies with hash (add more as required)
  // This method should be kept in sync with
  // org.apache.spark.network.util.JavaUtils#nonNegativeHash().
  private def nonNegativeHash(obj: AnyRef): Int = {

    // Required ?
    if (obj eq null) return 0

    val hash = obj.hashCode
    // math.abs fails for Int.MinValue
    val hashAbs = if (Int.MinValue != hash) math.abs(hash) else 0

    // Nothing else to guard against ?
    hashAbs
  }

  /**
   * Returns the hash index = term id = feature id = position in (Docs x Terms) matrix of the input term
   * by applying a a murmur hash on it
   */
  private def indexOf(term: String): Int = {

    //nonNegativeHash(term)
    //nonNegativeMod(term.##, numFeatures)
    Math.abs(MurmurHash3.stringHash(term)) % numFeatures
  }

  /**
   * Transforms the input document into a sparse term frequency vector.
   */
  private def transform(document: Iterable[String]): Vector = {
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    document.foreach { term =>
      val i = indexOf(term)
      // increase term frequency count
      termFrequencies.put(i, termFrequencies.getOrElse(i, 0.0) + 1.0)
    }

    // sorts by indices and creates indices array and values array
    // largest index has to be smaller than numFeatures!!
    Vectors.sparse(numFeatures, termFrequencies.toSeq)
  }

  /**
   * Transforms the input documents to term frequency vectors
   * I = document id
   * D = document as sequence over terms
   */
  def transform[D <: Iterable[String], I <: Product](dataset: RDD[(I, D)]): RDD[(I, Vector)] = {
    dataset.map(d => (d._1, this.transform(d._2)))
  }
}

