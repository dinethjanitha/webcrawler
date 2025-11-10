#Goolge search API
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv("./.env")

api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
cse_id = os.getenv("CUSTOM_SEARCH_ENGIN_ID")

def googlesearch(keyword):

    query = f'{keyword} site:facebook.com OR site:instagram.com OR site:linkedin.com OR site:twitter.com OR site:x.com OR site:youtube.com OR site:tiktok.com OR site:threads.net'

    try : 
        service = build("customsearch", "v1", developerKey=api_key)
        results = service.cse().list(
            q=query,
            cx=cse_id,
            num=5,  # Number of results to return (max 10)
            cr="sri lanka",
    ).execute()
    except Exception as e :
        print(e)

    if not results : 
        return None
    
    return results