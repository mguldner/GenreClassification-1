package de.tu_berlin.dima

import de.tu_berlin.dima.SVMParam.SVMParam
import scala.collection.JavaConversions._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{MultilabelMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vector
import net.sourceforge.argparse4j.ArgumentParsers

object SVMClassifierAnalysis {

  // svm params
  val REG = 0.0
  val NUM_IT = 10
  val STEP_SIZE = 1d

  case class ParamSearchResults(param: Double, precision: Double, recall: Double, accuracy: Double, hammingLoss: Double, f1Msr: Double) {}
  case class BinaryResults(genre: String, auROC: Double, auPR: Double)

  def main(args: Array[String]) {

    // get input arguments
    val ap = ArgumentParsers.newArgumentParser("ts")
    ap.addArgument("--movies").required(true)
    ap.addArgument("--out").required(true)
    ap.addArgument("--analysis").required(true)
    ap.addArgument("--param").required(false)
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
    val param = attrs.get("param").get match {
      case null => -1
      case n => n.toString.toInt
    }
    println(param)

    // set up exectution
    val conf = new SparkConf().setAppName("SVM Classifier")
    val sc = new SparkContext(conf)

    // get the genres
    val movies = PreProcessing.transformToMovieSynopsis(sc.textFile(moviesPath))
    val genres = movies.map(ms => ms.genre)
      .distinct()
      .collect()
      .toSeq
    println("genres: " + genres.toString)

    // pre process movie synopses to get tfidf representation of movie synopses
    val splits = PreProcessing.preProcess(movies)
    val tr = splits._1
    val ts = splits._2

    // count distribution of movies per genre for both, training and test sets
    println("Training: " + PreProcessing.moviesPerGenre(tr.map(t => t._1._3.head)).toString())
    println("Test: " + PreProcessing.moviesPerGenre(ts.flatMap(t => t._1._3)).toString())

    // run analysis
    if(analysisType.equals("binary")) {
      val svm = new SVMWithSGD()
      svm.optimizer
        .setNumIterations(NUM_IT)
        .setRegParam(REG)
        .setStepSize(STEP_SIZE)
      val res = oneVAllPerGenre(tr, ts, genres, svm)

      sc.parallelize(res
        .map(b => b.genre+","+b.auROC+","+b.auPR))
        .saveAsTextFile(outPath)
    } else if(analysisType.equals("paramsearch")) {
      require(param >= 0, "Param needs to be set: " + param)
      val p = SVMParam(param)
      val ps = getParamString(p)

      val res = paramSearch(tr, ts, genres, createParamList(param), p)

      sc.parallelize(res
        .map(r => ps+","+r.param+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
        .saveAsTextFile(outPath)
    } else if(analysisType.equals("multilabel")) {
      val res = paramSearch(tr, ts, genres, List(REG), SVMParam.REG)
      require(res.size == 1, "Multilabel only runs SVM once on each genre: " + res.toString())

      sc.parallelize(res
        .map(r => getParamString(SVMParam.REG)+","+REG+","+r.precision+","+r.recall+","+r.accuracy+","+r.hammingLoss+","+r.f1Msr))
        .saveAsTextFile(outPath)
    }

    sc.stop()
  }

  def getParamString(param: SVMParam): String = {
    param match {
      case SVMParam.REG => "Regularization"
      case SVMParam.NUM_IT => "Number of Iterations"
      case SVMParam.STEP_SIZE => "Step Size"
    }
  }

  def createParamList(param: Int): Seq[Double] = {
    SVMParam(param) match {
      case SVMParam.REG => List(0.0, 0.2, 0.6, 0.8)
      case SVMParam.NUM_IT => List(2,3,4,5,6,7,8,9,10)
      case SVMParam.STEP_SIZE => List(1, 2, 3, 4)
    }
  }

  /**
   * Run the classifier on different parameter values for one parameter
   * The parameter is either regularization or number of iterations for the SVM
   * @param train training set of tuples (documentId, tfidf-features)
   * @param test test set of tuples (documentId, tfidf-features)
   * @param genres genres that comprise the documents
   * @param params different parameter values
   * @param paramType indicate which param to optimize
   * @return List of different multilabel metrics for each parameter value
   */
  def paramSearch(train: RDD[((String, Int, Seq[String]), Vector)],
                  test: RDD[((String, Int, Seq[String]), Vector)],
                  genres: Seq[String],
                  params: Seq[Double],
                  paramType: SVMParam)
  : Seq[ParamSearchResults] =

  {
    var results = Seq[ParamSearchResults]()

    params.foreach(p => {
      println("Param: " + p)

      val metrics = oneVAll(train, test, genres, p, paramType)

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
   * @param param parameter value
   * @param paramType parameter type (regularization, number of iterations, step size)
   * @return evaluation metrics
   */
  def oneVAll(train: RDD[((String, Int, Seq[String]), Vector)],
              test: RDD[((String, Int, Seq[String]), Vector)],
              genres: Seq[String],
              param: Double,
              paramType: SVMParam)
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
      val svm = getSVM(param, paramType)

      // train the model
      val model = svm.run(trInput)

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
    println(m.labels.map(PreProcessing.doubleToGenre  ).toSeq)
    genres.foreach(g => {
      println(g)
      println("Prec:Rec")
      //println(m.precision(PreProcessing.genreToDouble(g)) + ":" + m.recall(PreProcessing.genreToDouble(g)))
    })

    m
  }

  /**
   * get an SVM analyser for the given parameter value
   * @param param parameter value
   * @param paramType parameter type
   * @return the SVM analyser
   */
  def getSVM(param: Double, paramType: SVMParam): SVMWithSGD = {

    val svm = new SVMWithSGD()

    paramType match {
      case SVMParam.REG => svm.optimizer.setRegParam(param).setNumIterations(NUM_IT).setStepSize(STEP_SIZE)
      case SVMParam.NUM_IT => svm.optimizer.setNumIterations(param.toInt).setRegParam(REG).setStepSize(STEP_SIZE)
      case SVMParam.STEP_SIZE => svm.optimizer.setStepSize(param).setRegParam(REG).setNumIterations(NUM_IT)
      case other => throw new RuntimeException("Wrong SVM Param: " + other)
    }

    svm
  }

  def oneVAllPerGenre(train: RDD[((String, Int, Seq[String]), Vector)],
                      test: RDD[((String, Int, Seq[String]), Vector)],
                      genres: Seq[String],
                      svm: SVMWithSGD)
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
      val model = svm.run(trInput)

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
