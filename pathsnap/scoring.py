"""Probability batching shared by training and reference-free inference."""
import numpy as np

def positive_scores(model,X,batch_size=100000):
    result=np.empty(len(X),np.float32)
    index=int(np.flatnonzero(model.classes_==1)[0])
    for lo in range(0,len(X),batch_size):
        result[lo:lo+batch_size]=model.predict_proba(X[lo:lo+batch_size])[:,index]
    return result
