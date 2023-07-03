import numpy as np
from collections import deque
from tqdm import tqdm

from scipy.linalg import toeplitz
from scipy.spatial import distance_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer

# FUNCTIONAL CLASS
class CLaSP():
    def __init__(self, window=10, cycles=3, solver='knn'):
        assert solver == 'knn' or solver == 'svd', 'Solver must be knn or svd'
        assert window % 2 == 0 , 'Window must be even'
        
        self.window = window
        self.cycles = cycles
        self.solver = solver
        self.stack = deque([])
    
    def fit(self, vec_ts, n_segments=3):
        # create X_dist and X
        self.create_X_and_distance_filtered(vec_ts)
        
        # initialize stack
        first_elem = {'start':0,
                      'stop':self.X_dist.shape[0],
                      'clasp_scores':None,
                      'clasp_max_idx':None,
                      'clasp_max_val':None}
        self.stack.append(first_elem)
        
        # while we don't have enough segments
        while len(self.stack) < n_segments:
            # calculate clasp on all elements
            for i in range(len(self.stack)):
                elem = self.stack.popleft()
                if elem['clasp_max_idx'] is not None:
                    self.stack.append(elem) # put it back
                if elem['clasp_max_idx'] is None:
                    # skip short segents
                    if elem['stop'] - elem['start'] < self.window*3+1:
                        elem['clasp_max_idx'] = -1
                        elem['clasp_max_val'] = 0
                        self.stack.append(elem)
                        continue
                    # selecting the slice and calculating clasp scores
                    X_dist_cur = self.X_dist[elem['start']:elem['stop']].T[elem['start']:elem['stop']]
                    print('X_dist_cur', X_dist_cur.shape)
                    clasp_scores_cur = self.clasp_single_run(X_dist_cur)
                    dividing_idx = np.argmax(clasp_scores_cur[self.window:-self.window]) + self.window
                    dividing_val = clasp_scores_cur[dividing_idx]
                    elem['clasp_max_idx'] = dividing_idx
                    elem['clasp_max_val'] = dividing_val
#                     elem['clasp_scores'] = clasp_scores_cur
                    self.stack.append(elem) # put it back
                
            # comparison of computed CLASP vectors before splitting
            max_val = 0
            max_elem = None
            max_idx = None
            for i in range(len(self.stack)):
                elem = self.stack[i]
                if elem['clasp_max_val'] > max_val:
                    max_val = elem['clasp_max_val']
                    max_elem, max_idx = elem, i
            print(max_idx)
            # remove element from deque stack
            del self.stack[max_idx]
                
            # creating and adding best element split to stack
            left_elem = {'start':max_elem['start'],
                         'stop':max_elem['start']+max_elem['clasp_max_idx'],
                         'clasp_scores':None,
                         'clasp_max_idx':None}
            right_elem = {'start':max_elem['start']+max_elem['clasp_max_idx'],
                         'stop':max_elem['stop'],
                         'clasp_scores':None,
                         'clasp_max_idx':None}
            self.stack.append(left_elem)
            self.stack.append(right_elem)
                
    # dataset creation
    def create_X_and_distance_filtered(self, vec_ts):
        # expand vec by window//2 to keep same dims
        vec_left_expand = np.insert(vec_ts, obj=0, values=vec_ts[self.window//2:0:-1])
        vec_both_expand = np.insert(vec_left_expand, obj=-1, values=vec_ts[-self.window//2:])
        # matrix creation
        X_list = []
        j = 0
        while j < vec_both_expand.shape[0]-self.window:
            X_list.append(vec_both_expand[j:j+self.window])
            j += 1
        X = np.array(X_list)
        print('X shape=', X.shape)
        # distance matrix
        X_dist = distance_matrix(X, X)
        print('X_dist shape=', X_dist.shape)

        # filter on the diagonal so close points wont be so close
        col = np.zeros(X.shape[0])
        row = np.zeros(X.shape[0])
        col[0:self.window//2] = np.max(X_dist)*2
        row[0:self.window//2] = np.max(X_dist)*2
        filtr = toeplitz(col, row)
        X_dist_filtered = X_dist + filtr
        # writing to class attributes
        self.X = X
        if self.solver == 'knn':
            self.X_dist = X_dist_filtered
        elif self.solver == 'svd':
            self.X_dist = X_dist
        else:
            pass
    
    # cross validation on single split of y
    def clasp_cross_val(self, X_dist, y, n_neighbors=3, n_splits=3, shuffle=False):
        scores = []
        kf = StratifiedKFold(n_splits=3, shuffle=shuffle)
        for i, (train_index, test_index) in enumerate(kf.split(np.arange(X_dist.shape[0]), y)):
            # knn model init
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
            # selecting train dist matrix
            X_train_double = X_dist[train_index].T[train_index]
            y_train = y[train_index]
            # selecting test dist matrix
            X_test_double = X_dist[train_index].T[test_index]
            y_test = y[test_index]
            # fitting model
            knn.fit(X_train_double, y_train)
            # testing model
            y_pred = knn.predict(X_test_double)
            score = roc_auc_score(y_test, y_pred)
            scores.append(score)
        return scores
    
    # SVD cross-val
    def clasp_svd(self, X_dist, split_idx):
        X_dist_left = X_dist[0:split_idx].T[0:split_idx]
        X_dist_right = X_dist[split_idx:].T[split_idx:]   

        pca_left = PCA(n_components=1)
        pca_left.fit(X_dist_left)
        X_dist_left_pca = pca_left.inverse_transform(pca_left.transform(X_dist_left))

        pca_right = PCA(n_components=1)
        pca_right.fit(X_dist_right)
        X_dist_right_pca = pca_right.inverse_transform(pca_right.transform(X_dist_right))

        # no need to normalize since we summ the error of the both halves of matrix
        score_left = np.linalg.norm(X_dist_left_pca - X_dist_left)/np.linalg.norm(X_dist_left)
        score_right = np.linalg.norm(X_dist_right_pca - X_dist_right)/np.linalg.norm(X_dist_right)
        return score_left + score_right
    
    # cycling over all y-splits
    def clasp_single_run(self, X_dist):
        clasp = []
        for i in tqdm(range(3, X_dist.shape[0]-3)):
            y = np.ones(X_dist.shape[0])
            y[0:i] = 0
            if self.solver == 'knn':
                scores = self.clasp_cross_val(X_dist, y, n_neighbors=3, n_splits=3, shuffle=True)
            elif self.solver == 'svd':
                scores = self.clasp_svd(X_dist, i)
            
            clasp.append(np.mean(scores))
        return clasp