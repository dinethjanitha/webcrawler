from model.keyword import keyword_collection
from model.siteData import siteDataCollection
from model.summary import summaryCollection
from bson import ObjectId
from fastapi.responses import JSONResponse 




async def getAllDetailsById(id):
    print(id)
    aggregate = [
    {
        '$match': {
            '_id': ObjectId(id)
        }
    }, {
        '$lookup': {
            'from': 'sitesData', 
            'localField': '_id', 
            'foreignField': 'keywordId', 
            'as': 'sitesData'
        }
    }, {
        '$lookup': {
            'from': 'summary', 
            'localField': '_id', 
            'foreignField': 'keywordId', 
            'as': 'summary'
        }
    }, {
        '$group': {
            '_id': '$_id', 
            'keyword': {
                '$first': '$keyword'
            }, 
            'urls': {
                '$first': '$sitesData.siteUrl'
            }, 
            
            'content': {
                '$first': '$sitesData.content'
            }, 
            'summary': {
                '$first': {
                    '$arrayElemAt': [
                        '$summary.summary', 0
                    ]
                }
            }
        }
    }
]

    try : 
         result = await keyword_collection.aggregate(aggregate).to_list(length=None)
    except Exception as e:
        print(e)
    print("Aggregate result")

    # for doc in result :
    #     doc["_id"] = str(doc["_id"])
    # print(result)
    return result[0]


async def getAllPreviousKeywords():
    
    try : 
        result = await keyword_collection.find().to_list(length=None)
    except Exception as e:
        print(e)
    # for doc in result :
    #     doc["_id"] = str(doc["_id"])
    print("result")
    print(result)
    return result



async def deletePreviousCrawl(id):
    try :
        await keyword_collection.delete_many({"_id" : ObjectId(id)})
        await siteDataCollection.delete_many({"keywordId" : ObjectId(id)})
        await summaryCollection.delete_many({"keywordId" : ObjectId(id)})
        
        return JSONResponse(
            status_code=200,
            content={"status" : "success"}
        )
    except Exception as e :
        print(e)
        return JSONResponse(
            status_code=400,
            content={"status" : "fail"}
        )
    