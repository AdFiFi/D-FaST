import pandas as pd
import json


class Recorder(object):
    def __init__(self):
        self.results = []
        self.avg = {}
        self.med = {}
        self.max = {}
        self.min = {}

    def add_record(self, result):
        self.results.append(result)

    def save(self, path):
        table = pd.DataFrame.from_records(self.results)
        self.avg = table.mean().to_dict()
        self.med = table.median().to_dict()
        self.max = table.max(axis=0).to_dict()
        self.min = table.min(axis=0).to_dict()
        js = json.dumps(self.__dict__)
        # with open(path, mode='w', encoding='utf-8') as f:
        #     f.write(js)

    def get_avg(self):
        return self.avg

