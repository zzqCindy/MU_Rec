import pymongo, math

db_list = ['Analytical', 'Anatomy','Anthropology','Chemicals','Disciplines', 'Diseases','Health',
        'Humanities','Information','Organisms', 'Phenomena', 'Psychiatry', 'Technology']

class Abstract(object):

    def __init__(self, top_list, label):
        self.top_list = top_list
        self.label = label
        self.client = pymongo.MongoClient('127.0.0.1', 27017)
        self.db = self.client.test
        self.abs_topic = self.db.abstract_topic

    def save(self):
        abstract = {"label": self.label}
        for i in range(0,10):
            abstract['topic%d'%i] = float(self.top_list.get(i,0))
        id = self.abs_topic.insert_one(abstract)
        print(id)

    def recom(self):
        select = sorted(self.top_list.items(),key=lambda item:item[1],reverse=True)
        selected_data = list(self.abs_topic.find({'label' : self.label,
                                     'topic%d'%select[0][0] : { '$gt' : math.floor(select[0][1]), '$lt' : math.ceil(select[0][1])},
                                     'topic%d'%select[1][0] : { '$gt' : math.floor(select[1][1]), '$lt' : math.ceil(select[1][1])},
                                     'topic%d'%select[2][0] : { '$gt' : math.floor(select[2][1]), '$lt' : math.ceil(select[2][1])}}))
        # recom_list计算欧氏距离并对应到publication _id 返回pub_list
        result = []
        for data in selected_data:
            sum = 0
            for i in range(0,10):
                sum += abs(self.top_list[i]-data['topic%d'%i])
            result.append(sum)
        idx = list(range(0,len(result)))
        selected_idx = [x for _,x in sorted(zip(result,idx))]
        final_id = [selected_data[idx]['_id'] for idx in selected_idx[:20]]
        coll = self.db['%s_publication' %db_list[self.label]]
        recom_list = list(coll.find({'_id':{'$in':final_id}},{'_id':0}))
        return recom_list

    def query_abstract(self):
        abstract = self.db.abstract_topic.find()
        return abstract
