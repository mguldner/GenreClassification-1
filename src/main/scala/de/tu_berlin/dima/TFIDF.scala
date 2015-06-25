package de.tu_berlin.dima

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF

/**
 * Created by mathieu on 18/06/15.
 */
object TFIDF {

  def apply(trainingSet: RDD[MovieSynopsis]): RDD[Vector] = {

    val documents : RDD[Seq[String]] = trainingSet.map{_.synopsis.toLowerCase().split("\\W+").toSeq filter(_.nonEmpty)}

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)

    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)

    return tfidf
  }
}
