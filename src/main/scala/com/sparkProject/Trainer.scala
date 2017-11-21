package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    
	import spark.implicits._


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

	// On charge le fivhier prepare
	val filteredDF = spark.read.parquet("/cal/homes/arastel/my_TP_Spark/data/prepared_trainingset")
	
	filteredDF.createOrReplaceTempView("parquetFile")
	/**val namesDF = spark.sql("SELECT * FROM parquetFile LIMIT 2")
	namesDF.map(attributes => "Name: " + attributes(0)).show()**/

    /** TF-IDF **/

	// On sépare les mots
	val tokenizer = new RegexTokenizer()
		.setPattern("\\W+")
		.setGaps(true)
		.setInputCol("text")
		.setOutputCol("tokens")

	//val regexTokenized = tokenizer.transform(filteredDF)

	// Ici, on enleve les conjonctions	

	val remover = new StopWordsRemover()
  		.setInputCol("tokens")
  		.setOutputCol("nostops")

	//val noStopWords = remover.transform(regexTokenized)

	//val cvModel: CountVectorizerModel = new CountVectorizer()

	// Calcul des occurences
	val cvModel = new CountVectorizer()
	  .setInputCol("nostops")
	  .setOutputCol("vectorized")

	//val vectorized = cvModel.fit(noStopWords).transform(noStopWords)

	// partie operante de la TF IDF

	val idfModel = new IDF()
		.setInputCol("vectorized")
		.setOutputCol("tfidf")

	//val rescaledData = idfModel.fit(vectorized).transform(vectorized)
	//rescaledData.select("nostops", "tfidf").show()


    /** VECTOR ASSEMBLER **/

	// gestion des variables categorielles

	val countriesIndexer = new StringIndexer()
  		.setInputCol("country2")
  		.setOutputCol("country_indexed")

	//val indexed = countriesIndexer.fit(rescaledData).transform(rescaledData)

	val currencyIndexer = new StringIndexer()
  		.setInputCol("currency2")
  		.setOutputCol("currency_indexed")

	//val indexed = currencyIndexer.fit(indexed).transform(indexed)
	
	// rassemblement de toutes les colonnes input
	val assembler1 = new VectorAssembler()
		.setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
		.setOutputCol("features")

    /** MODEL **/

	// parametrage du modele

	val lr = new LogisticRegression()
		.setElasticNetParam(0.0)
		.setFitIntercept(true)
		.setFeaturesCol("features")
		.setLabelCol("final_status")
		.setStandardization(true)
		.setPredictionCol("predictions")
		.setRawPredictionCol("raw_predictions")
		.setThresholds(Array(0.7,0.3))
		.setTol(1.0e-6)
		.setMaxIter(300)


    /** PIPELINE **/

	// executions successives

	val pipeline = new Pipeline().setStages(Array(tokenizer, remover, cvModel, idfModel, countriesIndexer, currencyIndexer, assembler1, lr))


    /** TRAINING AND GRID-SEARCH **/


	// 90 pourcents du dataset devient l ensemble d'apprentissage
	val Array(training, test) = filteredDF.randomSplit(Array(0.9, 0.1))

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF,(55.0 to 95.0 by 20.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

	// on fait applique le grid search avec cv
    val model = trainValidationSplit.fit(training)

    val predictionsDF = model.transform(test)

    predictionsDF.groupBy("final_status", "predictions").count.show()

	
	
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

	// obtention de la mesure de performance

    val accuracy = evaluator.evaluate(predictionsDF)
    println("Precision = " + (accuracy))
    println("Erreur = " + (1.0 - accuracy))


    //model.write.save("LR-KickStarter")

  }
}
