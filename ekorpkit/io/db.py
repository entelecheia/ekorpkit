import urllib.parse


def connect_mongo(
    collection, database="report", server="localhost", username=None, password=None
):
    from pymongo import MongoClient

    if username is None:
        uri = "mongodb://%s/%s" % (server, database)
    else:
        username = urllib.parse.quote_plus(username)
        password = urllib.parse.quote_plus(password)
        uri = "mongodb://%s:%s@%s/%s?authMechanism=SCRAM-SHA-1" % (
            username,
            password,
            server,
            database,
        )
    # print('connecting to MongoDB: {}'.format(server))
    return MongoClient(uri)[database][collection]
