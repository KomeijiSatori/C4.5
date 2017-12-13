import math
import functools
import copy
import json
from collections import Counter


class Node(object):
	def __init__(self):
		self.children = {} # {"attr1":Node1, "attr2":Node2, "attr3":Node3, attr:Node}
		self.split_attr = None # select the split attribute
		self.label = None
	def is_leaf(self):
		return len(self.children) == 0


# implement using: http://blog.csdn.net/xuxurui007/article/details/18045943
# return the attr name
def get_attr(data):
	# distribution: {"group_name1": {"group_val1": {"label1": x, "label2": y, ...}, "group_val2":...}, "group_name2": ...}
	distribution = {}
	# label_count: {"label1": x, "label2": y, ...}
	label_count = {}

	for single_data in data:
		attrs = single_data["attrs"]
		label = single_data["label"]
		# add to label_count
		if label_count.get(label):
			label_count[label] += 1
		else:
			label_count[label] = 1
		# add to distribution
		for group_name, group_val in attrs.items():
			if not distribution.get(group_name):
				distribution[group_name] = {}
			group_vals = distribution[group_name]
			if not group_vals.get(group_val):
				group_vals[group_val] = {}
			val_labels = group_vals[group_val]
			if val_labels.get(label):
				val_labels[label] += 1
			else:
				val_labels[label] = 1
	
	# calculate info_label
	info_label = 0
	all_count = len(data)
	for label, count in label_count.items():
		info_label += (-count) / all_count * math.log(count / all_count) / math.log(2)
	# calculate info_attrs, info_attrs: {"group_name1": x, "group_name2": y, ...}
	info_attrs = {}
	for group_name, group_vals in distribution.items():
		info_attrs[group_name] = 0
		for group_val, label_vals in group_vals.items():
			cur_attr_count = functools.reduce(lambda x, y: x + y, label_vals.values())
			cur_val = 0
			for label, count in label_vals.items():
				cur_val += -(count) / cur_attr_count * math.log(count / cur_attr_count) / math.log(2)
			cur_val *= cur_attr_count / all_count
			info_attrs[group_name] += cur_val
	# calculate gain_attrs, gain_attrs: {"group_name1": x, "group_name2": y, ...}
	gain_attrs = {}
	for group_name, val in info_attrs.items():
		gain_attrs[group_name] = info_label - val
	# calculate the split info, h_attrs: {"group_name1": x, "group_name2": y, ...}
	h_attrs = {}
	for group_name, group_vals in distribution.items():
		h_attrs[group_name] = 0
		attr_val = {}
		for attr, labels in group_vals.items():
			attr_val[attr] = functools.reduce(lambda x, y: x + y, labels.values())
		for attr, count in attr_val.items():
			h_attrs[group_name] += -(count) / all_count * math.log(count / all_count) / math.log(2)
	# calculate the IGR, igr_attrs: {"group_name1": x, "group_name2": y, ...}
	igr_attrs = {}
	for group_name, val in gain_attrs.items():
		if abs(h_attrs[group_name]) < 1e-6:
			return group_name
		igr_attrs[group_name] = val / h_attrs[group_name]
	return max(igr_attrs, key=igr_attrs.get)


def make_nodes(data):
	# single data: {"attrs": {"group_name1": val1, "group_name2": val2, ...}, "label": x}
	# data: [single_data1, single_data2, ..., single_datak]
	# datas: [data1, data2, ..., datan]
	root = Node()
	nodes = [root]
	datas = [data]

	while len(datas) > 0:
		cur_data = datas[0]
		cur_node = nodes[0]
		del datas[0]
		del nodes[0]
		if len(cur_data) == 0:
			# TODO: deal with no data
			pass
		else:
			left_attrs = cur_data[0]["attrs"].keys()
			if len(left_attrs) == 0:
				# if reach a leaf node
				# find the most common label
				labels= [x["label"] for x in cur_data]
				c = Counter(labels)
				cur_node.label = c.most_common(1)[0][0]
			else:
				# cut tree if all data has the same label
				labels = {x["label"] for x in cur_data}
				if len(labels) == 1:
					cur_node.label = cur_data[0]["label"]
					continue

				# else split the data by one attribute
				split_attr = get_attr(cur_data)
				cur_node.split_attr = split_attr
				attr_vals = set([x["attrs"][split_attr] for x in cur_data])
				for attr_val in attr_vals:
					next_data = copy.deepcopy(list(filter(lambda x:x["attrs"][split_attr] == attr_val, cur_data)))
					for ind in range(len(next_data)):					
						next_data[ind]["attrs"].pop(split_attr, None)
					datas.append(next_data)
					next_node = Node()
					cur_node.children[attr_val] = next_node
					nodes.append(next_node)
	return root


# predict the result using decision tree with data, if error, None will be present
# data: [{"group_name1": attr1, "group_name2": attr2, ...}, ...]
def predict(root, data):
	res = []
	for cur_data in data:
		cur_node = root
		try:
			while not cur_node.is_leaf():
				split_attr = cur_node.split_attr
				cur_attr = cur_data[split_attr]
				cur_node = cur_node.children[cur_attr]
			res.append(cur_node.label)
		except:
			res.append(None)
	return res


if __name__ == "__main__":
	with open("train_data.json") as fp:
		train_data = json.load(fp)
	with open("test_data.json") as fp:
		test_data = json.load(fp)
	root = make_nodes(train_data)

	expect_results = [x["label"] for x in test_data]
	predict_data = [x["attrs"] for x in test_data]
	predict_results = predict(root, predict_data)
	total = len(test_data)
	right = 0
	for ind in range(len(test_data)):
		if expect_results[ind] == predict_results[ind]:
			right += 1
	acc = right / total
	print("right:", right, "total:", total, "accuracy:", acc)

