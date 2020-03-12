
import numpy as np

def collect_statistic(datas, name):
    """
        Args:
            - datas: a list of dict, or a dict of dict
            - name: name of data
    """

    lens = []
    for i, key in enumerate(datas):
        if type(datas) == dict:
            data = datas[key]
        else:
            data = key
        context = data['context']
        lens.append(len(context))

    print("====== {} = ===== ".format(name))
    print("Total: {}  Length MIN/MAX/AVG: {}/{}/{:.4f}".format(
            len(lens),np.min(lens),np.max(lens),np.average(lens)))



