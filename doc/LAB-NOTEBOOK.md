# Lab Notebook

[Data Investigation](data-investigation.md)

**Ideas**:
- Single classifier: already know that it will result in very high performance, though it would be useful to have a baseline and a confusion matrix
- Hierarchical classifiers
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 24 attack types it is
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 4 attack categories it is. This approach would mesh better with the test set with its additional 14 types *iff* the test set has an attack type to attack category mapping. Even without that, it may prove to be more general, but certainly less specific...
  - binary on anomalous or normal, if anomalous then employ another classifier to determine if it's smurf or neptune or other (to weed out the most populous classes), if it's neither then employ a third classifier to decide which of the smaller classes it belongs to.

**A note on metrics**  
Precision, recall, and F1 metrics related below were measured across classes by weighting the performance of each class by the number of instances in that class in the respective test set. This scheme may seem to over-report performance numbers by virtue of drowning a number of labels with the performance on the big labels, but the even-weighted performance combination scheme suffers from performance measurements that are too coarse-grained on the under-represented classes. It's a decision between the lesser of two evils...

----------
  
**Baseline - Single classifier**

Pre-processed all data to remove annoying and meaningless period from end of every line, de-duplicated records in training set, and removed two illegal service='icmp' records from the test set.

Running a random forest over all of the features:
- 50 trees
- 20 max depth
- 80/20 train/test split

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RandomForestRunner --driver-memory 2g --master local[7] build\libs\kdd-1999-0-fat.jar data\kdd-unique.csv output/rf-base 50 20`

Performance on 20% test set ([metrics](../output/rf-base/metrics.csv), [matrix](../output/rf-base/matrix.csv)):
- Precision: 99.9510
- Recall: 99.9570
- F1: 99.9540

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RFTrainSave_Exp1 --driver-memory 3g --master local[7] build\libs\kdd-1999-0-fat.jar data\kdd-unique.csv data\kddcup-test.csv output/rf-base-test 50 20`

Performance on KDD99 test set ([metrics](../output/rf-base-test/metrics.csv), [matrix](../output/rf-base-test/matrix.csv)):
- Precision: 86.5159
- Recall: 91.9396
- F1: 89.1453

Comments: The classifier performs ridiculously well on the testing data drawn from the same distribution as the training data. And it still performs decently on the separate test data, though definitely worse overall. Of course the biggest contributor here is that there are 18,729 (out of 311,027) records in that set that are impossible to predict correctly because their classes are unobserved in the training data. Additionally, `guess_passwd` and `warezmaster` are both particularly difficult to predict in the test set.

**Binary Classifier - 100 trees**

In an effort to do better *anomaly* detection, I processed the training and test sets such that the complement of labels `normal` were all assigned to `anomaly`. I used this train and test set with a random forest:
- 100 trees
- 20 max depth

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RFTrainSave_Exp2 --driver-memory 3g --master local[7] build\libs\kdd-1999-0-fat.jar data\kddcup-train-bin.csv data\kddcup-test-bin.csv output/rf-bin-test 100 20`

Performance on KDD99 test set ([metrics](../output/rf-bin-test/metrics.csv), [matrix](../output/rf-bin-test/matrix.csv)):
- Precision: 94.5288
- Recall: 92.5270
- F1: 93.5172

Comments: Binarizing the data along with doubling the number of trees in the forest bumped the performance ~4% overall over the performance of the all classes baseline classifier. The largest source of error in the model, by far, is `normal` false positives.

**Binary classifier - 200 trees**

I re-configured my [code](src/main/java/collinm/kdd99/RFTrainSave_Exp2.java) to load the given labels and binarize them according to the scheme in the previous experiment. I also doubled the number of trees in the random forest:
- 200 trees
- 20 max depth

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RFTrainSave_Exp2 --driver-memory 3g --master local[7] build\libs\kdd-1999-0-fat.jar data\kddcup-train-bin.csv data\kddcup-test-bin.csv output/rf-bin200-test 200 20`

Performance on KDD99 test set ([metrics](../output/rf-bin200-test/metrics.csv), [matrix](../output/rf-bin200-test/matrix.csv)):
- Precision: 94.4938
- Recall: 92.4620
- F1: 93.4669

Comments: Overall performance dropped ever-so-slightly but this is probably more due to random fluctuations than meaningful performance changes. The real meat of this experiment was to generate triplets of `(actual class, binarized class, predicted binarized class)` to see which actual classes are making up all of the false positives from the prior experiment... except it didn't work properly and only the binarized output is showing up. The interaction of my UDF and Spark's `DataFrame.withColumn` function is not behaving as I expect, and there's pretty much zero documentation on either, so I'm going to abandon this line of coding for now and solve the problem with some post-processing.

I re-read some of my ideas from earlier and realized that implementing ensembled or hierarchical classifiers is going to be *very* painful in the Spark ML/LIB framework, especially since there doesn't seem to be an obvious or easy way to de/serialize random forest models. I suppose I'll direct my attention towards feature engineering then.

**Binary Classifier - Finding target class of false positives in test set**

Created a new [experiment](src/main/java/collinm/kdd99/RFTrainSave_Exp2_BinOutput.java) based on the prior binary classifier that trains on the binary data but classifies the *standard* test data. This way I can see, with some post-processing, which anomaly classes are attracting the most `normal` false positives.

Command to run pipeline: `spark-submit --class collinm.kdd99.RFTrainSave_Exp2_BinOutput --driver-memory 3g --master local[7] build\libs\kdd-1999-0-fat.jar data\kddcup-train-bin.csv data\kddcup-test.csv output/rf-bin-test-output 100 20`

Comments: I already know from a prior experiment that the largest source of error in the binary classifier is `normal` false positives. The following table shows the actual classes of instances from the test set that were incorrectly labeled with `normal` by the binary classifier. From this table, I excluded labels that had fewer than 20 incorrect instances (23 labels, 160 total instances). Lastly, I've highlighted labels that occur in the training data, the rest only occur in the test data.

| Attack Type | Labeled `normal` | Labeled `anomaly` |
| ----------- | ---------------- | ----------------- |
| snmpgetattack | 7741 | 0 |
| mailbomb | 5000 | 0 |
| **guess_passwd** | 4367 | 0 |
| snmpguess | 2406 | 0 |
| **warezmaster** | 1197 | 405 |
| processtable | 747 | 12 |
| mscan | 719 | 334 |
| apache2 | 630 | 164 |
| httptunnel | 144 | 14 |

The binary classifier did not correctly classify any of the `snmpgetattack`, `mailbomb`, or `snmpguess` instances, leading me to believe that these instances look fundamentally different than any other anomalous instances in the training set and/or they look a lot like `normal` instances. Similarly, `guess_passwd`, which *does* occur in the training set, was never correctly identified and probably looks different than the prior attacks of that type. The remainder of the labels are mostly unobserved in the training set and fared somewhat better, but none of them performed especially well. These results would seem to disprove the hypothesis that new attacks are derivative of old ones and thus share some aspect of their signatures in the given feature set. Of course, the base feature set could be improved by engineering features from combinations or comparisons of existing features, but this is greatly helped by having some domain knowledge.

In order to give an idea of where the binary classifier does well, I've included another table below detailing the actual classes of the instances that the classifier correctly labeled as `anomaly`. Similar to the last table, I've excluded labels that had less than 100 correctly labeled instances and highlighted the labels that occur in the training set.

| Attack Type | Labeled `anomaly` | Labeled `normal` |
| ----------- | ----------------- | ---------------- |
| **smurf** | 164091 | 0 |
| **neptune** | 57999 | 2 |
| **satan** | 1632 | 1 |
| **back** | 1092 | 6 |
| saint | 729 | 7 |
| **warezmaster** | 405 | 1197 |
| **portsweep** | 354 | 0 |
| mscan | 334 | 719 |
| **ipsweep** | 302 | 4 |
| apache2 | 164 | 630 |

Perhaps unsurprisingly, the classifier performs very well on well-represented classes that occurred in the training data, though it also manages to pick up almost all of `saint` and 30% of `mscan`. As a partial explanation of the latter behavior, a [prior confusion matrix](output/rf-base-test/matrix.csv) shows that the `saint` class was commonly predicted as the `satan` class, suggesting that the two have similar signatures. Additionally, `warezmaster` and `apache2` make another appearance by virtue of having many more instances overall in the test set than most of the other classes.
