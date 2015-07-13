package de.tu_berlin.dima

import de.tu_berlin.dima.SVMClassifierAnalysis.{BinaryResults, ParamSearchResults}
import net.sourceforge.argparse4j.ArgumentParsers
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, MultilabelMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scala.collection.JavaConversions._


/**
 * Created by oliver on 05.07.15.
 */
object BayesClassifierAnalysis {

  var LAMBDA = 1d

  def main(args: Array[String]) {

    val ap = ArgumentParsers.newArgumentParser("ts")
    ap.addArgument("--movies").required(true)
    ap.addArgument("--out").required(true)
    ap.addArgument("--analysis").required(true)
    ap.addArgument("--lambda").required(false)
    ap.addArgument("--genres").required(true)
    var attrs: Map[String, AnyRef] = null
    try {
      attrs = ap.parseArgs(args).getAttrs.toMap
    } catch {
      case e: Exception => throw new RuntimeException("usage: --movies --out --analyis: " + args.toSeq.toString)
    }
    println(attrs.toString)
    val moviesPath = attrs.get("movies").get.toString
    val outPath = attrs.get("out").get.toString
    val analysisType = attrs.get("analysis").get.toString
    val lambda = attrs.get("lambda").get match {
      case null => -1
      case n => n.toString.toDouble
    }
    var genresToFilter = attrs.get("genres").get.toString.split(",")
    println(genresToFilter.toSeq.toString)

    // set up exectution
    val conf = new SparkConf().setAppName("Bayes Classifier")
    val sc = new SparkContext(conf)

    // get the genres
    val movies = PreProcessing
      .filterGenres(PreProcessing.transformToMovieSynopsis(sc.textFile(moviesPath)), genresToFilter)
    val genres = movies
      .map(_.genre)
      .distinct()
      .collect()
    println("genres: " + genres.toString)

    // pre process movie synopses
    val splits = PreProcessing.preProcess(movies)
    val tr = splits._1
    val ts = splits._2

    // count distribution of movies per genre for both, training and test sets
    println("Training: " + PreProcessing.moviesPerGenre(tr.map(t => t._1._3.head)).toString())
    println("Test: " + PreProcessing.moviesPerGenre(ts.flatMap(t => t._1._3)).toString())

    // run analysis
    if(analysisType.equals("binary")) {
      val bayes = new NaiveBayes()
      bayes.setLambda(lambda)
      val res = oneVAllPerGenre(tr, ts, genres, bayes)
      res.foreach(b => println(b.genre+","+b.auROC+","+b.auPR))
      /*sc.parallelize(res
        .map(b => b.genre+","+b.auROC+","+b.auPR))
        .saveAsTextFile(outPath)*/
    } else if(analysisType.equals("paramsearch")) {

      val res = paramSearch(tr, ts, genres, List(0, 0.1, 0.3, 0.5, 0.7, 1, 3, 6, 9, 12))

      res.foreach(r => println("Lambda"+","+lambda+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
      /*sc.parallelize(res
        .map(r => "Lambda"+","+r.param+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
        .saveAsTextFile(outPath)*/

    } else if(analysisType.equals("multilabel")) {
      val res = paramSearch(tr, ts, genres, List(lambda))
      require(res.size == 1, "Multilabel only runs SVM once on each genre: " + res.toString())
      res.foreach(r => println("Lambda"+","+lambda+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
      /*sc.parallelize(res
        .map(r => "Lambda"+","+lambda+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
        .saveAsTextFile(outPath)*/
    }

    sc.stop()
  }

  /**
   * Run the classifier on different parameter values for one parameter
   * The parameter is either regularization or number of iterations for the SVM
   * @param train training set of tuples (documentId, tfidf-features)
   * @param test test set of tuples (documentId, tfidf-features)
   * @param genres genres that comprise the documents
   * @param params different parameter values
   * @return List of different multilabel metrics for each parameter value
   */
  def paramSearch(train: RDD[((String, Int, Seq[String]), Vector)],
                  test: RDD[((String, Int, Seq[String]), Vector)],
                  genres: Seq[String],
                  params: Seq[Double])
  : Seq[ParamSearchResults] =

  {
    var results = Seq[ParamSearchResults]()

    params.foreach(p => {
      println("Param: " + p)

      val metrics = oneVAll(train, test, genres, p)

      results +:= ParamSearchResults(p, metrics.precision, metrics.recall, metrics.accuracy, metrics.hammingLoss, metrics.f1Measure)
    })

    results
  }

  /**
   * A one versus all SVM approach to assign multiple labels to each document in the test set
   * The train and test sets are comprised by movies from the given genres
   * An SVM model is trained for each genre and evaluated on the test set
   * The evaluation results are being returned
   * @param train train set of (documentId, features)
   * @param test test set of (documentId, features)
   * @param genres genres that comprise the documents
   * @param lambda smoothing factor for naive bayes
   * @return evaluation metrics
   */
  def oneVAll(train: RDD[((String, Int, Seq[String]), Vector)],
              test: RDD[((String, Int, Seq[String]), Vector)],
              genres: Seq[String],
              lambda: Double)
  : MultilabelMetrics =
  {
    // initialize the predictions list for each element of the test set
    var tsInput = test.map(p => {

      (p, Seq[String]())
    })

    //TODO check if this is possible in parallel -> is it possible to use g like that?!
    //TODO g to broadcast?!
    genres.foreach(g => {
      println("Run " + g)

      // the current genre is our positive class, the rest is our negative class
      val trInput = train.map(p => {
        require(p._1._3.size == 1, "Element in Train has more than one genre " + p._1._3.toString())
        if(g.equals(p._1._3.head))
          LabeledPoint(1d, p._2)
        else
          LabeledPoint(0d, p._2)
      })

      // get svm model for current parameter
      val model = NaiveBayes.train(trInput, lambda)

      // train the model
      //val model = svm.run(trInput)

      println("Predict " + g)

      // predict on the test set
      tsInput = tsInput.map(point => {
        val score = model.predict(point._1._2)

        // make sure the score is not the raw margin
        require(score == 1.0 || score == 0.0, "Score is raw: " + score)

        // if movie is assigned to positive class, put it in predictions list
        if(score > 0)
          (point._1, point._2.:+(g))
        else
          point
      })
    })

    // transform test points to prediction and labels for Mulitlabel metric input
    val predLabls = tsInput.map(point => {
      val labels = point._1._1._3
      val predictions = point._2
      //if(predictions.nonEmpty) println("Pred:Label " + predictions.toString()+":"+labels.toString())

      (predictions.map(genre => PreProcessing.genreToDouble(genre)).toArray,
        labels.map(genre => PreProcessing.genreToDouble(genre)).toArray)
    })

    val m = new MultilabelMetrics(predLabls)
    //println(m.labels.map(PreProcessing.doubleToGenre  ).toSeq)
    genres.foreach(g => {
      println(g)
      println("Prec:Rec")
      println(m.precision(PreProcessing.genreToDouble(g)) + ":" + m.recall(PreProcessing.genreToDouble(g)))
    })

    m
  }

  def oneVAllPerGenre(train: RDD[((String, Int, Seq[String]), Vector)],
                      test: RDD[((String, Int, Seq[String]), Vector)],
                      genres: Seq[String],
                      bayes: NaiveBayes)
  : Seq[BinaryResults] =
  {
    var results = Seq[BinaryResults]()

    //TODO check if this is possible in parallel -> is it possible to use g like that?!
    //TODO maybe transform g to broadcast!!
    genres.foreach(g => {

      // the current genre is our positive class, the rest is our negative class
      val trInput = train.map(p => {
        require(p._1._3.size == 1, "Element in Train has more than one genre " + p._1._3.toString())
        if(g.equals(p._1._3.head))
          LabeledPoint(1d, p._2)
        else
          LabeledPoint(0d, p._2)
      })
      val tsInput = test.map(p => {
        if(p._1._3.contains(g))
          LabeledPoint(1d, p._2)
        else
          LabeledPoint(0d, p._2)
      })

      // train
      val model = bayes.run(trInput)

      // test
      val scoresAndLabels = tsInput.map(point => {
        val score = model.predict(point.features)

        // make sure the score is not the raw margin
        require(score == 1.0 || score == 0.0, "Score is raw: " + score)

        if(score > 0)
          (1d, point.label)
        else
          (0d, point.label)
      })

      // evaluate
      val metrics = new BinaryClassificationMetrics(scoresAndLabels)

      results +:= BinaryResults(g, metrics.areaUnderROC(), metrics.areaUnderPR())
    })

    results
  }
}
