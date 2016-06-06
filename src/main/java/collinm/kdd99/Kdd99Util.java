package collinm.kdd99;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import collinm.util.ConfusionMatrix;

public class Kdd99Util {

	public static DataFrame readData(Path filePath, SQLContext sql) {
		StructType kdd99Schema = new StructType(
				new StructField[] { new StructField("duration", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("protocol_type", DataTypes.StringType, false, Metadata.empty()),
						new StructField("service", DataTypes.StringType, false, Metadata.empty()),
						new StructField("flag", DataTypes.StringType, false, Metadata.empty()),
						new StructField("src_bytes", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("dst_bytes", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("land", DataTypes.StringType, false, Metadata.empty()),
						new StructField("wrong_fragment", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("urgent", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("hot", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_failed_logins", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("logged_in", DataTypes.StringType, false, Metadata.empty()),
						new StructField("num_compromised", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("root_shell", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("su_attempted", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_root", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_file_creations", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_shells", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_access_files", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("num_outbound_cmds", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("is_host_login", DataTypes.StringType, false, Metadata.empty()),
						new StructField("is_guest_login", DataTypes.StringType, false, Metadata.empty()),
						new StructField("count", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("srv_count", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("serror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("srv_serror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("rerror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("srv_rerror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("same_srv_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("diff_srv_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("srv_diff_host_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_count", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("dst_host_srv_count", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("dst_host_same_srv_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_diff_srv_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_same_src_port_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_srv_diff_host_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_serror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_srv_serror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_rerror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("dst_host_srv_rerror_rate", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("attack_type", DataTypes.StringType, false, Metadata.empty()) });

		DataFrame df = sql.read().format("com.databricks.spark.csv").schema(kdd99Schema).load(filePath.toString());
		return df;
	}

	/**
	 * Write precision, recall, and F1 for each matrix to a file.
	 * 
	 * @param outputDir
	 *            output directory
	 * @param matrices
	 *            confusion matrices to evaluate
	 */
	public static void writeMetrics(Path outputDir, ConfusionMatrix matrix) {
		try {
			Files.createDirectories(outputDir);
		} catch (IOException io) {
			System.out.println("Could not create output directory!");
			io.printStackTrace();
		}

		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputDir.toString(), "metrics.csv"))) {
			writer.write(ConfusionMatrix.toMetricsCSV(Arrays.asList(matrix)));
		} catch (IOException io) {
			System.out.println("Could not wite metrics out to file!");
			io.printStackTrace();
		}
	}

	/**
	 * Write out a confusion matrix to a file.
	 * 
	 * @param outputDir
	 *            output directory
	 * @param matrix
	 *            confusion matrix to write
	 * @param filename
	 *            target file
	 */
	public static void writeConfusionMatrix(Path outputDir, ConfusionMatrix matrix, String filename) {
		try {
			Files.createDirectories(outputDir);
		} catch (IOException io) {
			System.out.println("Could not create output directory!");
			io.printStackTrace();
		}

		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputDir.toString(), filename))) {
			writer.write(matrix.toCSV());
		} catch (IOException io) {
			System.out.println("Could not wite confusion matrix out to file!");
			io.printStackTrace();
		}
	}
	
	/**
	 * Make a mini-pipeline for converting a categorical variable in
	 * <code>colName</code> to a one-hot vector in <code>colName + "_vec"</code>
	 * .
	 * 
	 * @param colName
	 *            target column name
	 * @return Pipeline to perform the conversion
	 */
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
