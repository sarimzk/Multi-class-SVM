import numpy as np
import keras
from keras.datasets import mnist

def main():
    """Orchestrate the retrival of data, training and testing."""
    data = get_data()

    # Get classifier
    from sklearn.svm import SVC
    clf = SVC(probability=False,  # cache_size=200,
              kernel="linear", C=2.8, gamma=.0073)

    print("Start fitting. This may take a while")

    # take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

    analyze(clf, data)


def analyze(clf, data):
    """
    Analyze how well a classifier performs on data.

    Parameters
    ----------
    clf : classifier object
    data : dict
    """
    # Get confusion matrix
    from sklearn import metrics
    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))




def get_data():
    
    from sklearn.datasets import fetch_mldata
    from sklearn.utils import shuffle
    mnist = fetch_mldata('MNIST original')

    x = mnist.data
    y = mnist.target

    # Scale data to [-1, 1] - This is of mayor importance!!!
    x = x/255.0*2 - 1

    x, y = shuffle(x, y, random_state=0)

    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
                                                            
    data = {'train': {'X': x_train, 'y': y_train},  'test': {'X': x_test, 'y': y_test}}
                          
               
                         
    return data


if __name__ == '__main__':
    main()
