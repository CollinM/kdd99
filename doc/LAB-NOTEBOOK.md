# Lab Notebook

[Data Investigation](data-investigation.md)

**Ideas**:
- Single classifier: already know that it will result in very high performance, though it would be useful to have a baseline and a confusion matrix
- Hierarchical classifiers
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 24 attack types it is
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 4 attack categories it is. This approach would mesh better with the test set with its additional 14 types *iff* the test set has an attack type to attack category mapping. Even without that, it may prove to be more general, but certainly less specific...
  - binary on anomalous or normal, if anomalous then employ another classifier to determine if it's smurf or neptune or other (to weed out the most populous classes), if it's neither then employ a third classifier to decide which of the smaller classes it belongs to.
  
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

I re-configured my code to load the given labels and binarize them according to the scheme in the previous experiment. I also doubled the number of trees in the random forest:
- 200 trees
- 20 max depth

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RFTrainSave_Exp2 --driver-memory 3g --master local[7] build\libs\kdd-1999-0-fat.jar data\kddcup-train-bin.csv data\kddcup-test-bin.csv output/rf-bin200-test 200 20`

Performance on KDD99 test set ([metrics](../output/rf-bin200-test/metrics.csv), [matrix](../output/rf-bin200-test/matrix.csv)):
- Precision: 94.4938
- Recall: 92.4620
- F1: 93.4669

Comments: Overall performance dropped ever-so-slightly but this is probably more due to random fluctuations than meaningful performance changes. The real meat of this experiment was to generate triplets of `(actual class, binarized class, predicted binarized class)` to see which actual classes are making up all of the false positives from the prior experiment... except it didn't work properly and only the binarized output is showing up.
