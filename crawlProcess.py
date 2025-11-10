# from scrapy.crawler import CrawlerProcess
# from webscrapy.webscrapy.spiders.web_spider import WebSpider
# from webscrapy.webscrapy.spiders.web_spider_new import WebCrawSpider
# from connection.mongocon import mongoCon
from googlesearchmethod.googlesearch import googlesearch
# from scrapy import signals
# from pydispatch import dispatcher
from dotenv import load_dotenv
import os
from urllib.parse import urlparse , urlunparse
from datetime import datetime, timedelta

from fastapi import HTTPException

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from bson.objectid import ObjectId
from model.keyword import keyword_collection
from model.siteData import siteDataCollection
from model.summary import summaryCollection

from neo4j import GraphDatabase


import subprocess
import sys
import json
import re

load_dotenv("./env")

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# print(URI)
# print(AUTH)

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# Agent to access neo4j
@tool
def queryNeo4J(cypher_query:str) -> dict:
    
    """Get KG from Neo4j"""

    print("\n" + "=" * 80)
    print("STEP 1.*: Getting details from Neo4j...")
    print("=" * 80)

    # print("NeoStart")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
    return records


# Agent for make decision
@tool
def makeDecisionFromKG(query: str) -> str:
    """
    Ask the LLM to make a decision based on knowledge graph data.
    """
    reasoning_prompt = f"""
    You are an intelligent analyst with Neo4j.

    Question: {query}

    Analyze the relationships, infer insights, and give a concise, logical answer.
    """

    print("\n" + "=" * 80)
    print("STEP 1.*: Making Decision From Neo4j KG...")
    print("=" * 80)

    # print("In here nowwww")

    response = llm.invoke([HumanMessage(content=reasoning_prompt)])
    return response.content


async def ReasoningAgent():
    SYSTEM_PROMPT = """
    You are an intelligent AI reasoning agent connected to a Neo4j Knowledge Graph.
    Your capabilities:
    - Discover schema elements (labels, relationship types, property keys) when the user doesn't know exact KG keywords.
    - Generate Cypher queries that use fuzzy/partial matching to find relevant nodes and relationships.
    - Analyze query results and make decisions or summaries using the tool `makeDecisionFromKG`.
    - Link node by similarity and check again with that and find how relation in it neo4j if you look it as another type make query with it

    Tools available:
    1. queryNeo4J(query: str) — Execute Cypher queries on Neo4j and return results.
    2. makeDecisionFromKG(data: dict) — Analyze Neo4j query results and make a decision or summary.

    High-level rules:
    - Always start by discovering schema candidates relevant to the user's query (labels, relationship types, property keys) before issuing content queries.
    - NEVER hallucinate labels, relationship types, or properties that are not discoverable in the graph. Use actual results from Neo4j to decide.
    - Prefer safe, read-only Cypher (MATCH, RETURN, CALL db.*) unless explicitly asked to write.
    - Use fuzzy matching (`CONTAINS`, `toLower()`, or case-insensitive regex) when matching user terms to schema elements or data values.
    - If no matches are found, report that clearly and provide suggested alternative search terms, synonyms, or explain how the user could rephrase.

    Schema-discovery queries (Neo4j-native):
    - List all relationship types:
    CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType;
    - List all labels:
    CALL db.labels() YIELD label RETURN label;
    - List all property keys:
    CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey;

    Fuzzy-search templates (replace <term>):
    - Find relationship types matching a user term:
    MATCH ()-[r]-()
    WHERE toLower(type(r)) CONTAINS toLower('<term>')
    RETURN DISTINCT type(r) AS relType, count(r) AS occurrences
    ORDER BY occurrences DESC;
    - Find labels that match a user term:
    CALL db.labels() YIELD label
    WHERE toLower(label) CONTAINS toLower('<term>')
    RETURN label;
    - Find nodes whose properties match a user term:
    MATCH (n)
    WHERE any(k IN keys(n) WHERE toString(n[k]) =~ '(?i).*<term>.*')
    RETURN labels(n) AS labels, n AS node, size(keys(n)) AS propertyCount;

    Once a candidate relationship or label is found, fetch content nodes:
    MATCH (a)-[r:`<relationship>`]->(b)
    RETURN labels(a) AS fromLabels, a.name AS fromName,
        type(r) AS rel,
        labels(b) AS toLabels, b.name AS toName;

    When you find candidate relationship types or labels:
    - Return a short ranked list of best matches (relType or label, count of occurrences).
    - Automatically run a follow-up content query on the top candidates and summarize results using `makeDecisionFromKG`.

    Fallback behavior:
    - If no schema or data matches are found for the user term:
    - Return: "No matching labels or relationship types found for '<term>' in the knowledge graph."
    - Provide 2–4 suggested synonyms or alternate search terms the user might try.
    - Suggest an explicit schema-discovery run (CALL db.* queries) if permitted by the user.

    Safety and precision:
    - Always put the user term into safe, parameterized Cypher or escape user input properly to avoid syntax issues.
    - Prefer `toLower(... ) CONTAINS toLower(...)` for robust partial matching. Use regex `=~ '(?i).*term.*'` only when needed.

    Response style:
    - Be clear, structured, and logical.
    - For schema discovery steps, show the query used and the succinct ranked results (up to 5 candidates).
    - For content queries, summarize findings and pass the raw results to `makeDecisionFromKG` for final interpretation.
    """

    tools = [queryNeo4J, makeDecisionFromKG]

    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=InMemorySaver()
    )
    return agent


async def test_decision(keywordId: str , user_prompt:str):
    # Initialize reasoning agent
    agent = await ReasoningAgent()

    print("\n" + "=" * 80)
    print("STEP 1: Start Agent...")
    print("=" * 80)

    # Prepare user query
    user_message = f"""
    Retrieve data about keywordId '{keywordId}' and decide:
    Task: {user_prompt}

    """

    improved_user_message = f"""
    Task: {user_prompt}

    Data Retrieval Instruction:
    1. Retrieve data about associated with the internal parameter keywordId: '{keywordId}'.
    2. Analyze the retrieved performance data.
    3. Execute the Task described above.
    4. Never mention keywordId in your final output.

    Output Format: Provide the final response, including the analysis and decision, in Markdown (.md) format.
    """

    # print(user_message)
    # Call the agent
    print("Generating Cypher Query for access knowledge graph... ")

    result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": improved_user_message}
        ]
    },
    config={"configurable": {"thread_id": "thread_1"}
            } 
    )

    # Safely extract output
    output = result.get("output") or result.get("text") or str(result)

    messages_list = result.get("messages", [])

    print("\n" + "=" * 80)
    print("STEP 2: Checking Agent result...")
    print("=" * 80)

    final_content = None
    if messages_list:
        # Get the last message object from the list
        last_message = messages_list[-1]
        
        # Get the actual text content from that message object
        final_content = last_message.content
        print(final_content)
        
        # print("Decision:\n", final_content[0]["text"])

        print("\n" + "=" * 80)
        print("STEP 3: Finalizing...")
        print("=" * 80)
        try:
            # 1. Get the list of messages.
            # The output key is often 'messages', but could be 'output' or 'chat_history'.
            if "messages" in result:
                messages_list = result["messages"]
            elif "output" in result and isinstance(result["output"], list):
                messages_list = result["output"]
            else:
                print("Could not find a 'messages' list in the result.")
                print("Full result keys:", result.keys())
                # Set an empty list to avoid crashing later
                messages_list = []

            # 2. Check if the list is not empty
            if messages_list:
                # Get the last message object
                last_message = messages_list[-1]
                
                # 3. Get the .content attribute
                content = last_message.content
                
                final_text = ""
                
                # 4. Check the type of content and extract text
                if isinstance(content, list) and content:
                    # It's a list, get the 'text' from the first dictionary
                    final_text = content[0].get('text', 'No "text" key found in content dict')
                
                elif isinstance(content, str):
                    final_text = content
                
                else:
                    final_text = str(content) # Convert to string as a fallback

                print("--- Final AI Message ---")
                print(final_text)
                
                return {
                    "status" : "success",
                    "message" : final_text
                }
            else:
                print("No messages found in the list.")

                return HTTPException(status_code=404,  detail={
                    "status" : "fail",
                    "details" : "Somethings wrong check terminal for find error" 
                })

        except Exception as e:
            print(f"An error occurred: {e}")
            print("--- Full Agent Result for Debugging ---")
            print(result)
            return HTTPException(status_code=404, detail={
                "status" : "fail",
                "details" : "Somethings wrong check terminal for find error" 
            })
    else:
        # This helps you debug if the agent's output format is different
        print("Error: Could not find 'messages' in the result.")
        print("Full result:", result)
        return  HTTPException(status_code=404,  detail={
                    "status" : "fail",
                    "details" : "Somethings wrong check terminal for find error" 
                })

    # print("Decision:\n", output['messages'].content)



@tool
async def getCrawlContent(keywordId:str) -> str:
    
    """Fetch crawl text data by keyword ID (string). Returns all combined text content for that keyword."""

    print("\n" + "=" * 80)
    print("STEP 5.*: Getting crawling content from database...")
    print("=" * 80)

    now = datetime.utcnow()
    ten_minutes_ago = now - timedelta(minutes=10)
    
    siteDataResults = await siteDataCollection.find({'keywordId' : ObjectId(keywordId) , 'createdAt': {'$gte': ten_minutes_ago}  }).to_list(length=None)
    
    content = []
    for document in siteDataResults:
        content.append(document['content'])
    print("content")
    print(len(content))
    if len(content) > 0 :
        joinAllContent = "".join(content)
        print(f"Total content length: {len(joinAllContent)} characters")
        return joinAllContent
    else :
        return ""
    

@tool
def createKG(content:str , keywordId:str) -> object:
    """After get crawl content create Knowledge Graph and return Knowledge Graph JSON format """

    print("\n" + "=" * 80)
    print("STEP 5.*: Creating Knowledge Graph...")
    print("=" * 80)

    prompt_template = """
    You are an expert in extracting structured knowledge from text.

    Input: {crawl_text}

    Task:
    - Identify all nodes (entities) and relationships (edges) mentioned in the text.
    - Output ONLY valid JSON in this format:
    - All letters should be simple letters 

    {{
    "nodes": [  
        {{
        "label": "<NodeLabel>",
        "name": "<NodeName>",
        "properties": {{"key": "value"}}
        }}
    ],
    "edges": [
        {{
        "from": "<SourceNodeName>",
        "type": "<RelationType>",
        "to": "<TargetNodeName>",
        "properties": {{"key": "value"}}
        }}
    ]
    }}
    """


    prompt = PromptTemplate(
        input_variables=["crawl_text"],
        template=prompt_template,
    )

    full_prompt = prompt.format_prompt(
        crawl_text=content
    )

    try:
        print("Generating JSON schema for create knowledge graph... ")
        llm_response = llm.invoke(full_prompt)
    
        clean_text = re.sub(r"^```json\s*|\s*```$", "", llm_response.content.strip())

        json_out = json.loads(clean_text)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error!")
    
    print(llm_response.content)

    saveKGToNeo4j(keywordId , json_out)
    return json_out


def saveKGToNeo4j(keywordId: str, kg_json: dict):
    print("\n" + "=" * 80)
    print("STEP 5.*: Saving KG in Neo4j...")
    print("=" * 80)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            try:
                # Delete old graph for this keyword
                session.run("MATCH (n {keywordId: $id}) DETACH DELETE n", {"id": keywordId})

                # Create all nodes
                for node in kg_json["nodes"]:
                    label = node["label"]
                    name = node["name"]
                    properties = node.get("properties", {})
                    properties.update({"name": name, "keywordId": keywordId})
                    prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                    session.run(f"CREATE (n:{label} {{ {prop_str} }})", properties)

                # Create relationships
                for edge in kg_json["edges"]:
                    rel_type = re.sub(r"[^A-Za-z0-9_]", "_", edge["type"]).upper()
                    props = edge.get("properties", {})
                    props["keywordId"] = keywordId
                    props["from"] = edge["from"]
                    props["to"] = edge["to"]

                    session.run(f"""
                        MATCH (a {{name: $from, keywordId: $keywordId}}),
                              (b {{name: $to, keywordId: $keywordId}})
                        CREATE (a)-[r:{rel_type} {{keywordId: $keywordId}}]->(b)
                    """, props)

            except Exception as e:
                print(" Neo4j error:", e)
                raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")


async def MyAgent():
    SYSTEM_PROMPT = """
    You are an intelligent agent that can gather crawl data by keyword and create knowledge graphs automatically.
    You have access to two tools:
    - getCrawlContent: fetches all crawl text for a given keyword ID.
    - createKG: converts raw text into a structured knowledge graph.
    """

    checkpointer = InMemorySaver()
    tools = [getCrawlContent, createKG]

    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        # checkpointer=checkpointer
    )


    return agent

# Run Agent
async def FullAutoAgent(keywordId: str):

    print("\n" + "=" * 80)
    print("STEP 5.1: Calling Agents")
    print("=" * 80)
    agent_executor = await MyAgent()

    print("keywordId")
    print(keywordId)

    # Step 1 + 2 + 3: Crawl content → Create KG
    response = await agent_executor.ainvoke(
    {
        "messages": [
            {"role": "user", "content": f"Generate a knowledge graph for keyword ID {keywordId}"}
        ]
    },
    config={"configurable": {"thread_id": "kg_1"}}
    )


    # Step 4: Save to Neo4j
    print("Knowledge Graph saved to Neo4j successfully.")

    return response


# Stored Keyword in mongoDB
async def storeKeyword(keyword , url_list):

    if  not url_list or len(url_list) == 0:
        mydict = {
            "keyword" : keyword,
            "urls" : url_list,
        } 

    else : 
        mydict = {
            "keyword" : keyword,
        }

    
    try:
        x = await keyword_collection.insert_one(mydict) 
        print("---x----") 
        print(x) 
    except Exception as e:
        print(e)
        return None    
    print("xxxxxxxxxxxxxxxxxxxxxx")
    print(x)
    return x


# Get details with keyword ID
async def getKeywordById(id):
    try:
        result = await keyword_collection.find_one({"_id" : ObjectId(id)})
    except Exception as e:
        print(e)
        return None    
    return result

# Get details with keyword name
async def getKeywordByDomain(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "") 

        result = await keyword_collection.find_one(
            {"keyword": {"$regex": domain, "$options": "i"}}
        )

        return result
        
    except Exception as e:
        print(e)
        return None    
    return result


# Add urls to keyword document
async def storeRelevantUrls(keywordId , urls_list):
    
    try:
        keywordDetails = await getKeywordById(keywordId)
        
        keyword = keywordDetails["keyword"]
        # siteDomain = keywordDetails["siteDomain"]

        # results = googlesearch(keyword , siteDomain)

        # urlList = []

        # for item in results.get("items", []):
        #     print(f"Title: {item['title']}")
        #     urlList.append(item['link'])
        #     print(f"Link: {item['link']}\n")

        print(urls_list)

        updatedValues = await keyword_collection.update_one(
            {"_id": ObjectId(keywordId)},
            {"$push": {"urls": {"$each": urls_list}}}
        )
        print("Updated Values")
        print(updatedValues)

        if updatedValues.acknowledged:
            print("Update successful!")
            result = keywordId
            return result    
        return None
    except Exception as e:
        print(e)
        return None


# Crawl web data using subprocess
async def crawlUrls(urls, keywordId):
    """
    Runs the web crawler in a separate subprocess
    Returns: True if successful, False if failed
    """
    python_path = os.path.join(sys.prefix, "Scripts", "python.exe")  # Windows venv
    
    if not os.path.exists(python_path):
        python_path = os.path.join(sys.prefix, "bin", "python")  # Linux/Mac
    
    print("=" * 80)
    print("Starting crawler subprocess")
    print(f"Keyword ID: {keywordId}")
    print(f"Total URLs to crawl: {len(urls)}")
    print("=" * 80)
    
    try:
        # Run web_crawl_runner.py with URLs and keywordId as arguments
        process = subprocess.run(
            [python_path, "web_crawl_runner.py", *urls, str(keywordId)],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=300  # 5 minutes timeout
        )
        
        print("\n--- Crawler Output ---")
        print(process.stdout)
        
        if process.stderr:
            print("\n--- Crawler Warnings/Errors ---")
            print(process.stderr)
        
        print(f"\n--- Return Code: {process.returncode} ---")
        
        if process.returncode == 0:
            print("SUCCESS: Crawler completed successfully!")
            return True
        else:
            print(f"FAILED: Crawler failed with exit code {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Crawler timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: Subprocess exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def summarizeUsingAgent(keywordId):

    joinAllContent = None

    print(keywordId)
    try:
        keywordDetails = await keyword_collection.find_one({'_id' : ObjectId(keywordId)})

        siteDataResults = await siteDataCollection.find({'keywordId' : ObjectId(keywordId)}).to_list(length=None)
        print("siteDataResults")
        mainKeyword = keywordDetails['keyword']
        print("mainKeyword")
        print(mainKeyword)
        content = []
        for document in siteDataResults:
            content.append(document['content'])

        print("content")
        print(len(content))
        if len(content) > 0 :
            joinAllContent = "".join(content)
            print(f"Total content length: {len(joinAllContent)} characters")

        openai_key = os.getenv("GOOGLE_API_KEY")

        

        prompt = f"Summarize the following and align that details with this keyword {mainKeyword} **this summarize get word crawl result so mention it in top and not top as provide text show it as crawl we summary results** (using .md style to your response): {joinAllContent if joinAllContent else 'No text found'}"

        print("Prompt length: ", len(prompt))

        

        response = llm.invoke([HumanMessage(content=prompt)])


        # Print the result's content
        print("Summary generated successfully!")
        print(response.content) 

        summaryData = {"keywordId" : ObjectId(keywordId) , "summary" : response.content }

        await summaryCollection.insert_one(summaryData)

        return response.content
    except Exception as e:
        print(f"Summarization error: {e}")
        return None

async def exec(keyword , url_list):
    """
    Complete workflow:
    1. Store keyword
    2. Fetch Google search URLs
    3. Crawl URLs (subprocess)
    4. Summarize content (only if crawl succeeds)
    """
    
    # Step 1: Store keyword or add it to existing keyword
    print("\n" + "=" * 80)
    print("STEP 1.1: Check keyword")
    print("=" * 80)
    
    
        
    result = await getKeywordByDomain(keyword)
    skipSum = False
    if not result : 
        print("\n" + "=" * 80)
        print("STEP 1.2: Storing keyword")
        print("=" * 80)
        storedKeyword = await storeKeyword(keyword, url_list)
        storedKeywordId = storedKeyword.inserted_id
        print(f"Keyword stored with ID: {storedKeywordId}")

        # if url_list and len(url_list) > 0:
        #     updatedKey = await storeRelevantUrls(storedKeyword.inserted_id , None)
            
    else : 
        print("Id is founded!")
        print(result["_id"])
        storedKeywordId = result["_id"]
        skipSum = True
        print("Keyword Already founded! Skip creating new keyword id...")
    # Step 2: Get keyword details
    print("\n" + "=" * 80)
    print("STEP 2: Fetching keyword details")
    print("=" * 80)
    resultMongo = await getKeywordById(storedKeywordId)
    keywordId = resultMongo["_id"]
    # Step 3: Fetch Google URLs
    # print("\n" + "=" * 80)
    # print("STEP 3: Fetching Google search URLs")
    # print("=" * 80)
    # updatedKey = await storeRelevantUrls(storedKeyword.inserted_id)
    
    # if not keywordId:
    #     print("ERROR: Failed to store URLs")
    #     return {"error": "Failed to fetch URLs from Google"}
    
    # Get updated details with URLs
    print("\n" + "=" * 80)
    print("STEP 3: Checking keyword details")
    print("=" * 80)
    updatedDetails = await getKeywordById(keywordId)
    
    # if "urls" not in updatedDetails or not updatedDetails["urls"]:
    #     print("ERROR: No URLs found!")
    #     return {"error": "No URLs found in Google search results"}
    
    url = updatedDetails["keyword"]
    # url_list = updatedDetails["urls"]
    urls = [url]
    if url_list and len(url_list) > 0:
        print("\n" + "=" * 80)
        print("STEP 3.1: Manual added url fined crawling started with it!")
        print("=" * 80)
        urls += url_list
    
    print(f"Found URL {updatedDetails["keyword"]}  URLs to crawl")
    # for i, url in enumerate(urls, 1):
    #     print(f"   [{i}] {url}")

    # Step 4: Crawl URLs
    print("\n" + "=" * 80)
    print("STEP 4: Starting web crawl")
    print("=" * 80)
    
    crawl_success = await crawlUrls(urls, keywordId)
    
    if not crawl_success:
        print("ERROR: Crawl failed!")
        return {
            "error": "Web crawl failed",
            "keyword_id": str(keywordId),
            "urls_attempted": len(urls)
        }
    

    print("\n" + "=" * 80)
    print("STEP 5: Start Agentic AI")
    print("=" * 80)

    resultAgent = await FullAutoAgent(keywordId)

    print("------------------------\n Result Agent\n------------------------")
    print(resultAgent)
    # Step 5: Summarize (only if crawl succeeded)
    print("\n" + "=" * 80)
    print("STEP 6: Generating AI summary")
    print("=" * 80)
    

    if skipSum == False : 
        finalValue = await summarizeUsingAgent(keywordId)
        if finalValue == None :
            return {
            "status": "Summarization failed!",
        }
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            "status": "success",
            "keyword_id": str(keywordId),
            "urls_crawled": len(urls),
            "urls" : urls,
            "summary": finalValue
        }
    else : 
         return {
            "status": "success",
            "keyword_id": str(keywordId),
            # "urls_crawled": len(urls),
            # "urls" : urls,
            # "summary": finalValue
        }