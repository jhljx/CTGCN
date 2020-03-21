import os
import pandas as pd
import numpy as np

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)

class MATHransformer:
    input_file: str
    format_output_path: str
    use_good_data: bool
    good_data_format: str
    good_data_list: list

    def __init__(self, input_file, output_base_path):
        self.input_file = input_file
        self.format_output_path = os.path.join(output_base_path, "1.format")
        self.node_list_output_path = os.path.join(output_base_path, "nodes_set")

        # 创建路径
        check_and_make_path(self.format_output_path)

    def transform(self, trans_type=None, use_good_data=False):
        print("transforming MATH...")

        self.use_good_data = use_good_data

        if trans_type is None:
            trans_type = ['month', 'year']

        if 'month' in trans_type:
            self.handle_by_month()

        if 'year' in trans_type:
            self.handle_by_year()

        print("transforming MATH complete\n")

    def handle_by_year(self):
        # 按年处理
        dataframe = pd.read_csv(self.input_file, sep=" ", names=['from_id', 'to_id', 'timestamp'])
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y')
        candidate = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
        good_data = ['2010', '2011', '2012', '2013', '2014', '2015']
        if self.use_good_data:
            self.good_data_format = '%Y'
            self.good_data_list = good_data
        for year in (good_data if self.use_good_data else candidate):
            tem = dataframe[['from_id', 'to_id']][dataframe['timestamp'] == year]
            tem['from_id'] = tem['from_id'].map(lambda x: "U" + str(x))  # user
            tem['to_id'] = tem['to_id'].map(lambda x: "U" + str(x))  # user
            # 统计权重
            tem = tem.groupby(['from_id', 'to_id']).size().reset_index().rename(columns={0: 'weight'})
            tem.to_csv(os.path.join(self.format_output_path, str(year) + ".csv"), sep='\t', header=1, index=0)

    def handle_by_month(self):
        # 按月处理
        dataframe = pd.read_csv(self.input_file, sep=" ", names=['from_id', 'to_id', 'timestamp'])
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y-%m')
        candidate = ['2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04', '2010-05',
                     '2010-06', '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02',
                     '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11',
                     '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08',
                     '2012-09', '2012-10', '2012-11', '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05',
                     '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02',
                     '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11',
                     '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08',
                     '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03']
        good_data = ['2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09',
                     '2012-10', '2012-11', '2012-12']
        good_data = ['2009-10', '2009-11', '2009-12', '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06',
                     '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12', '2011-01', '2011-02', '2011-03',
                     '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12',
                     '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09',
                     '2012-10', '2012-11', '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06',
                     '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03',
                     '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12',
                     '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09',
                     '2015-10', '2015-11', '2015-12', '2016-01', '2016-02']
        if self.use_good_data:
            self.good_data_format = '%Y-%m'
            self.good_data_list = good_data
        for month in (good_data if self.use_good_data else candidate):
            tem = dataframe[['from_id', 'to_id']][dataframe['timestamp'] == month]
            tem['from_id'] = tem['from_id'].map(lambda x: "U" + str(x))  # user
            tem['to_id'] = tem['to_id'].map(lambda x: "U" + str(x))  # user
            # 统计权重
            tem = tem.groupby(['from_id', 'to_id']).size().reset_index().rename(columns={0: 'weight'})
            tem.to_csv(os.path.join(self.format_output_path, str(month) + ".csv"), sep='\t', header=1, index=0)

    def test_granularity(self, time_format='%Y-%m-%d %H:%M:%s'):
        dataframe = pd.read_csv(self.input_file, sep=" ", names=['from_id', 'to_id', 'timestamp'])
        print("top 10 rows:\n")
        print(dataframe[0:10])
        print("shape:")
        print(dataframe.shape)
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime(time_format)
        print(np.sort(dataframe['timestamp'].unique()))

    def get_full_node_set(self):
        print("get full node set")

        nodes_set_path = self.node_list_output_path
        check_and_make_path(nodes_set_path)

        dataframe = pd.read_csv(self.input_file, sep=" ", names=['from_id', 'to_id', 'timestamp'])

        if self.use_good_data:
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime(self.good_data_format)
            dataframe = dataframe[np.isin(dataframe['timestamp'], self.good_data_list, invert=False)]

        # nodes_set = np.sort(pd.concat([dataframe['from_id'], dataframe['to_id']], axis=0, ignore_index=True).unique())
        nodes_set = pd.concat([dataframe['from_id'], dataframe['to_id']], axis=0, ignore_index=True).unique()
        pd_data = pd.DataFrame(nodes_set, columns=['node'])
        pd_data['node'] = pd_data['node'].map(lambda x: "U" + str(x))
        pd_data = pd_data.sort_values(by='node')

        pd_data.to_csv(os.path.join(nodes_set_path, 'nodes.csv'), index=0, header=0)

        print("got it\n")

def transform(trans_type=None, use_good_data=False):
    if trans_type is None:
        trans_type = ['month']
    math = MATHransformer(input_file="/data/math/0.input/sx-mathoverflow.txt", output_base_path="/data/math")
    math.transform(trans_type=trans_type, use_good_data=use_good_data)
    math.get_full_node_set()

if __name__ == '__main__':
    transform(trans_type=['month'], use_good_data=True)