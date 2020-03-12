
import numpy as np

def collect_statistic(data, name):
    """
        Args:
            - data: dict
            - name: name of data
    """

    # 1. lenghts
    lens = []
    for key, item in data.items():
        if "question" in item:
            lens.append(len(item["question"]))
        elif "context" in item:
            lens.append(len(item["context"]))
        else:
            pass
    
    print("====== {} = ===== ".format(name))
    print("Total: {}  Length MIN/MAX/AVG: {}/{}/{:.4f}".format(
            len(lens),np.min(lens),np.max(lens),np.average(lens)))



