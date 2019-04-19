import pymongo

# 返回一个 collection
def get_coll():

    client = pymongo.MongoClient('127.0.0.1', 27017)
    db = client.test
    abs_topic = db.abstract_topic

    return abs_topic


class Abstract(object):

    def __init__(self, content, top_list):
        self.content = content
        self.top_list = top_list

    def save(self):
        abstract = {"content": self.content}
        for i in range(0,10):
            abstract['topic%d'%i] = float(self.top_list.get(i,0))
        print(abstract)
        coll = get_coll()
        id = coll.insert_one(abstract)
        print(id)

    @staticmethod
    def query_abstract():
        abstract = get_coll().find()
        return abstract
