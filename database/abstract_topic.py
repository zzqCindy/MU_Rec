import pymongo, math

# 返回一个 collection
def get_db():

    client = pymongo.MongoClient('127.0.0.1', 27017)
    db = client.test

    return db


class Abstract(object):

    def __init__(self, content, top_list):
        self.content = content
        self.top_list = top_list
        self.db = get_db()

    def save(self):
        abstract = {"content": self.content}
        for i in range(0,10):
            abstract['topic%d'%i] = float(self.top_list.get(i,0))
        abs_topic = self.db.abstract_topic
        id = abs_topic.insert_one(abstract)
        print(id)

    def recom(self):
        select = sorted(self.top_list.items(),key=lambda item:item[1],reverse=True)
        abs_topic = self.db.abstract_topic
        recom_list = abs_topic.find({'topic%d'%select[0][0] : { '$gt' : math.floor(select[0][1]), '$lt' : math.ceil(select[0][1])},
                                     'topic%d'%select[1][0] : { '$gt' : math.floor(select[1][1]), '$lt' : math.ceil(select[1][1])},
                                     'topic%d'%select[2][0] : { '$gt' : math.floor(select[2][1]), '$lt' : math.ceil(select[2][1])}})
        # recom_list计算欧氏距离并对应到publication _id 返回pub_list



        return recom_list

    def query_abstract(self):
        abstract = self.db.abstract_topic.find()
        return abstract
