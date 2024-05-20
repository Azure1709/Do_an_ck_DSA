from __future__ import print_function 
import numpy as np 
import pandas as pd 

class TreeNode(object):
    """
    Thông tin của các nút trên Decision Tree
    ids                 chỉ số của các dữ liệu trong nút
    entropy             entropy của nút này
    depth               độ sâu của nút
    split_attribute     thuộc tính đã chọn để tách nút, nút này không phải nút lá
    children            tập các nút con của nút này
    order               thứ tự giá trị của split_attribute trong các nút con
    label               nhãn của nút nếu nó là nút lá
    """
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids
        self.entropy = entropy
        self.depth = depth
        self.split_attribute = None
        self.children = children
        self.order = None
        self.label = None

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    """
    Tính giá trị Entropy
    Lưu ý rằng chúng ta đang quy ước 0log0 = 0, nên ta cần xóa các xác xuất bằng 0
    """
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    """
    Chứa thông tin của Decision Tree:
    root                gốc của cây
    max_depth           độ sâu tối đa của cây
    min_sample_split    số điểm dữ liệu tối thiểu có trong một nút
    Ntrain              số dữ liệu huấn luyện
    min_gain            lượng entropy giảm tối thiểu của một phép phân chia
    """
    def __init__(self, max_depth = 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        """
        Huấn luyện cây dựa trên tập dữ liệu data và tập nhãn target của nó
        """
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)

        self.target = target 
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children: # nút lá
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, ids):
        """
        Tính entropy của các điểm dữ liệu ids 
        Lưu ý rằng pandas series value_counts tính chỉ số bắt đầu từ 1
        """
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids]
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        """
        Find label for a node if it is a leaf
        Chose by major voting
        Gán nhãn cho nút nếu nó là nút lá
        Chọn nhãn có nhiều nhất trong nút đó
        """ 
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0]) # most frequent label
    
    def _split(self, node):
        """
        Tách node thành các node con
        """
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue # entropy = 0
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
            # Không phân chia nếu có một nút có số điểm dữ liệu quá nhỏ
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            HxS= 0              
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS 
            if gain < self.min_gain: continue # không chia nếu information gain quá nhỏ
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split, entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """
        new_data: tập dữ liệu mới, mỗi hàng là một điểm dữ liệu
        return: dự đoán nhãn cho mỗi điểm dữ liệu
        """
        npoints = new_data.count()[0]
        labels = [None]*npoints
        for i in range(npoints):
            x = new_data.iloc[i, :] # lấy một điểm dữ liệu
            # bắt đầu từ gốc và đi đến lúc gặp nút lá
            node = self.root
            while node.children: 
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[i] = node.label
            
        return labels

if __name__ == "__main__":
    df = pd.read_csv('data.csv', index_col = 0, parse_dates = True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTreeID3(max_depth = 10, min_samples_split = 2)

    pf = pd.read_csv('data_predict.csv', index_col = 0, parse_dates = True)
    X_predict = pf.iloc[:, :]

    tree.fit(X, y)
    y_predict = tree.predict(X_predict)
    
    print(y_predict)