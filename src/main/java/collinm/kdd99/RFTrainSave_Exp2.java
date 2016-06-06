package collinm.kdd99;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.javatuples.Triplet;

import collinm.util.ConfusionMatrix;

public class RFTrainSave_Exp2 {
	
	/**
	 * 
	 * @param args
	 *            <code>train-file test-file output-directory num-trees max-depth</code>
	 */
	public static void main(String[] args) {
		Path trainFile = Paths.get(args[0]);
		Path testFile = Paths.get(args[1]);
		Path outputDir = Paths.get(args[2]);
		int trees = Integer.parseInt(args[3]);
		int depth = Integer.parseInt(args[4]);

		// Setup Spark
		SparkConf conf = new SparkConf().setAppName("RandomForest");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sql = new SQLContext(jsc);
		// Register binarizing function
		sql.udf().register("binarizer", new UDF1<String, String>() {
			private static final long serialVersionUID = 1L;
			public String call(String label) {
				return label.equals("normal") ? "normal" : "anomaly";
			}
		}, DataTypes.StringType);

		// Read data
		System.out.println("Reading in data");
		DataFrame train = Kdd99Util.readData(trainFile, sql);
		DataFrame test = Kdd99Util.readData(testFile, sql);
		train = train.withColumn("attack_type_bin", functions.callUDF("binarizer", train.col("attack_type")));
		test = test.withColumn("attack_type_bin", functions.callUDF("binarizer", test.col("attack_type")));
		DataFrame all = train.unionAll(test);
		
		// Transform attack type into index ahead of time due to extra classes being present in the test data
		StringIndexer targetIndexer = new StringIndexer()
				.setInputCol("attack_type_bin")
				.setOutputCol("attack_type_index");
		StringIndexerModel targetIndexerModel = targetIndexer.fit(all);
		train = targetIndexerModel.transform(train);
		
		// Create pipeline
		IndexToString targetUnIndexer = new IndexToString()
				.setInputCol("predicted_attack_type_index")
				.setOutputCol("predicted_attack_type_label")
				.setLabels(targetIndexerModel.labels());
		Pipeline pipe1 = Kdd99Util.makeOneHotEncodedPipeline("protocol_type");
		Pipeline pipe2 = Kdd99Util.makeOneHotEncodedPipeline("service");
		Pipeline pipe3 = Kdd99Util.makeOneHotEncodedPipeline("flag");
		Pipeline pipe4 = Kdd99Util.makeOneHotEncodedPipeline("land");
		Pipeline pipe5 = Kdd99Util.makeOneHotEncodedPipeline("logged_in");
		Pipeline pipe6 = Kdd99Util.makeOneHotEncodedPipeline("is_host_login");
		Pipeline pipe7 = Kdd99Util.makeOneHotEncodedPipeline("is_guest_login");
		VectorAssembler featureAssembler = new VectorAssembler()
				.setInputCols(new String[] { "duration", "protocol_type_vec", "service_vec", "flag_vec", "src_bytes",
						"dst_bytes", "land_vec", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in_vec",
						"num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
						"num_access_files", "num_outbound_cmds", "is_host_login_vec", "is_guest_login_vec", "count",
						"srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
						"same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
						"dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
						"dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
						"dst_host_rerror_rate" })
				.setOutputCol("features");
		RandomForestClassifier rfc = new RandomForestClassifier()
				.setNumTrees(trees)
				.setMaxDepth(depth)
				.setFeatureSubsetStrategy("auto")
				.setLabelCol("attack_type_index")
				.setPredictionCol("predicted_attack_type_index");
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] {pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe7, featureAssembler, rfc, targetUnIndexer});
		
		// Train model
		PipelineModel model = pipeline.fit(train);
		train.unpersist();
		
		// Evaluate test data
		test.persist();
		test = targetIndexerModel.transform(test);
		DataFrame predxns = model.transform(test);
		ConfusionMatrix metrics = new ConfusionMatrix(Kdd99Util.BIN_CLASSES);
		List<Triplet<String, String, String>> perfTrips = new ArrayList<>();
		for (Row r : predxns.select("attack_type", "attack_type_bin", "predicted_attack_type_label").collect()) {
			Triplet<String, String, String> trip = Triplet.with(r.getString(0), r.getString(1), r.getString(2));
			perfTrips.add(trip);
			metrics.increment(trip.getValue1(), trip.getValue2());
		}
		test.unpersist();

		// Write metrics
		Kdd99Util.writeMetrics(outputDir, metrics);
		Kdd99Util.writeConfusionMatrix(outputDir, metrics, "matrix.csv");
		Kdd99Util.writeTriplets(perfTrips, outputDir);
		
		
		// Finish
		jsc.close();
		System.out.println("FINISHED!");
	}
}
