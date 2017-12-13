## C4.5 implementation
Build the decision tree using C4.5 algorithm, while providing a predict method to see the accuracy.
### Data Format
1. The training data should be named as `train_data.json`, and the test data should be named as `test_data.json`.
2. Both the training data and test data are of same json structure, as described below:
```
[{"attr": {"attr_name1": attr1, "attr_name2": attr2, ..., "attr_namek": attrk}, "label": label}, ...]
```
Where `attr_name1`, `attr_name2`, ... are all attribute names, the names should be identical, but you are free to choose whatever attribute name you like.

And `attr1`, `attr2`, ... are all value of `attr_name`.

The `label` refer to the output value of the input.

If you are not quite sure, please refer to `train_example.json` and `test_example.json`

