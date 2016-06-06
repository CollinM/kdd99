# Lab Notebook

[Data Investigation](data-investigation.md)

**Ideas**:
- Single classifier: already know that it will result in very high performance, though it would be useful to have a baseline and a confusion matrix
- Hierarchical classifiers
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 24 attack types it is
  - binary on anomalous or normal, if anomalous then employ another classifier to determine which of the 4 attack categories it is. This approach would mesh better with the test set with its additional 14 types *iff* the test set has an attack type to attack category mapping. Even without that, it may prove to be more general, but certainly less specific...
  - binary on anomalous or normal, if anomalous then employ another classifier to determine if it's smurf or neptune or other (to weed out the most populous classes), if it's neither then employ a third classifier to decide which of the smaller classes it belongs to.
  
**Baseline - Single classifier**

Pre-processed all data to remove annoying and meaningless period from end of every line.

Running a random forest over all of the features:
- 50 trees
- 20 max depth
- 80/20 train/test split

Command to run random forest pipeline: `spark-submit --class collinm.kdd99.RandomForestRunner --driver-memory 2g --master local[7] build\libs\kdd-1999-0-fat.jar data\kdd-unique.csv output/rf-base 50 20`

Performance on 20% test set ([metrics](../output/rf-bfase/metrics.csv), [matrix](../output/rf-bfase/matrix.csv):
- Precision: 99.9510
- Recall: 99.9570
- F1: 99.9540

