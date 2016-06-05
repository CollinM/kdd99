package collinm.kdd99;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import collinm.util.ConfusionMatrix;

public class RandomForestRunner {

	public final static String[] CLASSES = new String[] { "back", "buffer_overflow", "ftp_write", "guess_passwd",
			"imap", "ipsweep", "land", "loadmodule", "multihop", "neptune", "nmap", "perl", "phf", "pod", "portsweep",
			"rootkit", "satan", "smurf", "spy", "teardrop", "warezclient", "warezmaster", "normal" };
	
	/**
	 * 
	 * @param args
	 *            <code>input-file output-directory num-trees max-depth</code>
	 */
	public static void main(String[] args) {
		Path inputFile = Paths.get(args[0]);
		Path outputDir = Paths.get(args[1]);
		int trees = Integer.parseInt(args[2]);
		int depth = Integer.parseInt(args[3]);

		// Setup Spark
		SparkConf conf = new SparkConf().setAppName("RandomForest");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sql = new SQLContext(jsc);

		// Read data
		System.out.println("Reading in data");
		DataFrame wholeDf = Kdd99Util.readData(inputFile, sql);
		DataFrame[] trainTest = wholeDf.randomSplit(new double[] {0.8, 0.2}, 42L);
		trainTest[0].cache();
		trainTest[1].cache();
		
		// Create pipeline
		Pipeline pipe1 = makeOneHotEncodedPipeline("protocol_type");
		Pipeline pipe2 = makeOneHotEncodedPipeline("service");
		Pipeline pipe3 = makeOneHotEncodedPipeline("flag");
		Pipeline pipe4 = makeOneHotEncodedPipeline("land");
		Pipeline pipe5 = makeOneHotEncodedPipeline("logged_in");
		Pipeline pipe6 = makeOneHotEncodedPipeline("is_host_login");
		Pipeline pipe7 = makeOneHotEncodedPipeline("is_guest_login");
		StringIndexer targetIndexer = new StringIndexer()
				.setInputCol("attack_type")
				.setOutputCol("attack_type_index");
		StringIndexerModel targetIndexerModel = targetIndexer.fit(trainTest[0]);
		IndexToString targetUnIndexer = new IndexToString()
				.setInputCol("predicted_attack_type_index")
				.setOutputCol("predicted_attack_type_label")
				.setLabels(targetIndexerModel.labels());
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
				.setStages(new PipelineStage[] {pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe7, targetIndexer, featureAssembler, rfc, targetUnIndexer});
		
		// Train model
		PipelineModel model = pipeline.fit(trainTest[0]);
		
		// Evaluate model
		DataFrame predxns = model.transform(trainTest[1]);
		ConfusionMatrix metrics = new ConfusionMatrix(CLASSES);
		for (Row r : predxns.select("attack_type", "predicted_attack_type_label").collect())
			metrics.increment(r.getString(0), r.getString(1));
		
		// Write metrics
		Kdd99Util.writeMetrics(outputDir, metrics);
		Kdd99Util.writeConfusionMatrix(outputDir, metrics, "matrix.csv");
		
		// Finish
		jsc.close();
		System.out.println("FINISHED!");
	}
	
	public static Pipeline makeOneHotEncodedPipeline(String colName) {
		StringIndexer vi = new StringIndexer()
				.setInputCol(colName)
				.setOutputCol(colName + "_index");
		OneHotEncoder ohe = new OneHotEncoder()
				.setInputCol(colName + "_index")
				.setOutputCol(colName + "_vec");
		return new Pipeline().setStages(new PipelineStage[] {vi, ohe});
	}

}
