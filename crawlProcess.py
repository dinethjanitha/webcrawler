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


# Configuration for text chunking
MAX_CHUNK_SIZE = 5000  # Characters per chunk (adjust based on LLM token limit)
CHUNK_OVERLAP = 500     # Overlap between chunks to maintain context


# Error tracking for model/agent errors
error_log = []

def trackError(component: str, error_type: str, error_message: str, keywordId: str = None, details: dict = None):
    """
    Track errors that occur during model/agent execution
    
    Args:
        component: Where the error occurred (e.g., 'createKG', 'FullAutoAgent', 'LLM')
        error_type: Type of error (e.g., 'JSONParseError', 'ValidationError', 'TimeoutError')
        error_message: The error message
        keywordId: Associated keyword ID if applicable
        details: Additional details about the error
    """
    error_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": component,
        "error_type": error_type,
        "error_message": str(error_message),
        "keywordId": keywordId,
        "details": details or {}
    }
    
    error_log.append(error_entry)
    
    # Print formatted error
    print("\n" + "🔴" * 40)
    print(f" ERROR TRACKED:")
    print(f"   Component: {component}")
    print(f"   Type: {error_type}")
    print(f"   Message: {error_message}")
    if keywordId:
        print(f"   Keyword ID: {keywordId}")
    if details:
        print(f"   Details: {json.dumps(details, indent=2)}")
    print("🔴" * 40 + "\n")
    
    return error_entry


def getErrorLog(component: str = None, keywordId: str = None):
    """
    Retrieve error logs with optional filtering
    
    Args:
        component: Filter by component name
        keywordId: Filter by keyword ID
    
    Returns:
        List of error entries
    """
    filtered_errors = error_log
    
    if component:
        filtered_errors = [e for e in filtered_errors if e["component"] == component]
    
    if keywordId:
        filtered_errors = [e for e in filtered_errors if e["keywordId"] == keywordId]
    
    return filtered_errors


def getErrorSummary():
    """
    Get a summary of all tracked errors
    
    Returns:
        Dictionary with error statistics and recent errors
    """
    if not error_log:
        return {
            "total_errors": 0,
            "message": "No errors tracked"
        }
    
    # Count by component
    component_counts = {}
    error_type_counts = {}
    
    for error in error_log:
        comp = error["component"]
        err_type = error["error_type"]
        
        component_counts[comp] = component_counts.get(comp, 0) + 1
        error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
    
    return {
        "total_errors": len(error_log),
        "errors_by_component": component_counts,
        "errors_by_type": error_type_counts,
        "recent_errors": error_log[-5:],  # Last 5 errors
        "all_errors": error_log
    }


def chunkText(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split large text into smaller chunks with overlap
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    print(f"   📝 Chunking {text_length} chars into chunks of {chunk_size} with {overlap} overlap")
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If not the last chunk, try to break at a sentence or word boundary
        if end < text_length:
            # Look for sentence end (., !, ?)
            last_period = text.rfind('.', start, end)
            last_exclaim = text.rfind('!', start, end)
            last_question = text.rfind('?', start, end)
            
            sentence_end = max(last_period, last_exclaim, last_question)
            
            if sentence_end > start + (chunk_size // 2):  # If found in latter half
                end = sentence_end + 1
            else:
                # Fall back to word boundary
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            print(f"      Chunk {len(chunks)}: chars {start}-{end} ({len(chunk)} chars)")
        
        # Move start position with overlap
        if end >= text_length:
            break
        start = end - overlap
    
    print(f"    Created {len(chunks)} chunks")
    return chunks


def mergeKGJsons(kg_list: list) -> dict:
    """
    Merge multiple KG JSONs into one, removing duplicates
    
    Args:
        kg_list: List of KG JSON dictionaries
    
    Returns:
        Merged KG JSON with unique nodes and edges
    """
    merged_nodes = []
    merged_edges = []
    
    seen_nodes = set()  # Track unique nodes by (label, name)
    seen_edges = set()  # Track unique edges by (from, type, to)
    
    for kg in kg_list:
        # Merge nodes
        for node in kg.get("nodes", []):
            node_key = (node.get("label", ""), node.get("name", ""))
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                merged_nodes.append(node)
        
        # Merge edges
        for edge in kg.get("edges", []):
            edge_key = (edge.get("from", ""), edge.get("type", ""), edge.get("to", ""))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                merged_edges.append(edge)
    
    return {
        "nodes": merged_nodes,
        "edges": merged_edges
    }


def processChunkToKG(chunk_content: str, keywordId: str, chunk_num: int, total_chunks: int) -> dict:
    """
    Process a single chunk of content to create a partial KG
    
    Args:
        chunk_content: Text content to process
        keywordId: Keyword ID for error tracking
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
    
    Returns:
        KG JSON dictionary with nodes and edges
    """
    prompt_template = """
    You are an expert in extracting structured knowledge from text.
    Double check it and make it correctly

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

    full_prompt = prompt.format_prompt(crawl_text=chunk_content)
    
    print(f"      📤 Sending chunk {chunk_num}/{total_chunks} to LLM ({len(chunk_content)} chars)")
    print(f"         Preview: {chunk_content[:150]}...")

    try:
        llm_response = llm.invoke(full_prompt)
        print(f"      📥 Received LLM response ({len(llm_response.content)} chars)")
        clean_text = re.sub(r"^```json\s*|\s*```$", "", llm_response.content.strip())
        json_out = json.loads(clean_text)
        
        # Validate the JSON structure
        if "nodes" not in json_out or "edges" not in json_out:
            error_msg = "LLM response missing 'nodes' or 'edges' keys"
            trackError(
                component="processChunkToKG",
                error_type="InvalidJSONStructure",
                error_message=error_msg,
                keywordId=keywordId,
                details={
                    "chunk_num": chunk_num,
                    "total_chunks": total_chunks,
                    "llm_response": llm_response.content[:500],
                    "parsed_json": json_out
                }
            )
            raise ValueError(error_msg)
        
        return json_out
        
    except json.JSONDecodeError as e:
        error_details = {
            "chunk_num": chunk_num,
            "total_chunks": total_chunks,
            "llm_response": llm_response.content if 'llm_response' in locals() else "No response",
            "cleaned_text": clean_text if 'clean_text' in locals() else "No cleaned text",
            "parse_error": str(e),
            "content_preview": chunk_content[:200] if chunk_content else "No content",
            "content_length": len(chunk_content) if chunk_content else 0
        }
        
        trackError(
            component="processChunkToKG",
            error_type="JSONParseError",
            error_message=f"Failed to parse LLM response as JSON: {str(e)}",
            keywordId=keywordId,
            details=error_details
        )
        
        raise
    
    except Exception as e:
        trackError(
            component="processChunkToKG",
            error_type=type(e).__name__,
            error_message=str(e),
            keywordId=keywordId,
            details={
                "chunk_num": chunk_num,
                "total_chunks": total_chunks,
                "content_length": len(chunk_content) if chunk_content else 0
            }
        )
        raise


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
    ten_minutes_ago = now - timedelta(minutes=6)
    
    siteDataResults = await siteDataCollection.find({
        'keywordId': ObjectId(keywordId),
        'createdAt': {'$gte': ten_minutes_ago, '$lte': now}
    }).to_list(None)


    content = []
    for document in siteDataResults:
        if 'content' in document and document['content']:
            content.append(str(document['content']))
    
    print(f"Found {len(content)} documents in database")
    
    if len(content) > 0:
        # Join all content from all documents
        joinAllContent = "".join(content)
        content_length = len(joinAllContent)
        print(f"Total content length: {content_length} characters")
        print(f"   Preview (first 200 chars): {joinAllContent[:200]}...")
        
        # Check if content needs chunking - if so, process it here
        if content_length > MAX_CHUNK_SIZE:
            print(f"Content exceeds {MAX_CHUNK_SIZE} chars - chunking and processing here")
            
            # Create chunks
            chunks = chunkText(joinAllContent, MAX_CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Process each chunk and collect partial KGs
            all_partial_kgs = []
            
            for i, chunk in enumerate(chunks):
                print(f"\n   Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                
                try:
                    partial_kg = processChunkToKG(chunk, keywordId, i+1, len(chunks))
                    if partial_kg and (partial_kg.get("nodes") or partial_kg.get("edges")):
                        all_partial_kgs.append(partial_kg)
                        print(f"Chunk {i+1}: {len(partial_kg.get('nodes', []))} nodes, {len(partial_kg.get('edges', []))} edges")
                    else:
                        print(f"    Chunk {i+1}: No KG data extracted")
                        
                except Exception as e:
                    error_msg = f"Failed to process chunk {i+1}/{len(chunks)}: {str(e)}"
                    print(f"    {error_msg}")
                    trackError(
                        component="getCrawlContent",
                        error_type="ChunkProcessingError",
                        error_message=error_msg,
                        keywordId=keywordId,
                        details={
                            "chunk_number": i+1,
                            "total_chunks": len(chunks),
                            "chunk_length": len(chunk),
                            "error": str(e)
                        }
                    )
                    # Continue with other chunks even if one fails
                    continue
            
            # Merge all partial KGs and return as JSON string
            if all_partial_kgs:
                print(f"\n    Merging {len(all_partial_kgs)} partial knowledge graphs...")
                merged_kg = mergeKGJsons(all_partial_kgs)
                print(f"    Final merged KG: {len(merged_kg.get('nodes', []))} nodes, {len(merged_kg.get('edges', []))} edges")
                
                # Save to Neo4j immediately after merging
                print(f"\n    Saving merged KG to Neo4j...")
                try:
                    saveKGToNeo4j(keywordId, merged_kg)
                    print(f"    Successfully saved to Neo4j!")
                except Exception as e:
                    print(f"    Failed to save to Neo4j: {str(e)}")
                    trackError(
                        component="getCrawlContent->saveKGToNeo4j",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        keywordId=keywordId,
                        details={
                            "nodes_count": len(merged_kg.get("nodes", [])),
                            "edges_count": len(merged_kg.get("edges", []))
                        }
                    )
                
                # Return JSON string representation that createKG can handle
                return json.dumps({
                    "already_processed": True,
                    "kg_data": merged_kg
                })
            else:
                print(f"    All chunks failed to produce valid KG data")
                return json.dumps({
                    "already_processed": True,
                    "kg_data": {"nodes": [], "edges": [], "error": "All chunks failed"}
                })
        else:
            print(f" Content size OK ({content_length} chars) - returning for normal processing")
            return joinAllContent
    else:
        print(" No content found in database")
        return ""
    

@tool
def createKG(content:str , keywordId:str) -> object:
    """After get crawl content, create Knowledge Graph and return Knowledge Graph JSON format Double check and make it correctly """

    print("\n" + "=" * 80)
    print("STEP 5.*: Creating Knowledge Graph (AGENT TOOL)")
    print("=" * 80)
    print(f"🤖 Agent called createKG tool for keywordId: {keywordId}")
    
    # Check if content was already processed in chunks by getCrawlContent
    try:
        parsed_content = json.loads(content)
        if isinstance(parsed_content, dict) and parsed_content.get("already_processed"):
            print(" Content was already chunked, processed, and saved by getCrawlContent")
            json_out = parsed_content.get("kg_data", {"nodes": [], "edges": []})
            
            if json_out.get("nodes") or json_out.get("edges"):
                print(f" KG already saved in Neo4j: {len(json_out.get('nodes', []))} nodes, {len(json_out.get('edges', []))} edges")
                return json_out
            else:
                print(" Pre-processed KG is empty")
                return json_out
    except (json.JSONDecodeError, TypeError):
        # Not pre-processed JSON, continue with normal flow
        pass
    
    # Validate content before processing
    if not content or len(content.strip()) < 10:
        error_msg = f" Content is empty or too short (length: {len(content) if content else 0})"
        print(error_msg)
        trackError(
            component="createKG",
            error_type="EmptyContentError",
            error_message=error_msg,
            keywordId=keywordId,
            details={"content_length": len(content) if content else 0}
        )
        # Return empty KG structure instead of crashing
        return {
            "nodes": [],
            "edges": [],
            "error": "No content available to create knowledge graph"
        }
    
    content_length = len(content)
    print(f" Processing content: {content_length} characters")
    print(f"   First 200 chars: {content[:200]}...")
    
    # Process directly (content is small enough)
    print(f"   Content size OK - processing without chunking")
    try:
        json_out = processChunkToKG(content, keywordId, 1, 1)
        print(f" KG JSON validated: {len(json_out.get('nodes', []))} nodes, {len(json_out.get('edges', []))} edges")
    except Exception as e:
        error_msg = f"Failed to process content: {str(e)}"
        print(f" {error_msg}")
        trackError(
            component="createKG",
            error_type=type(e).__name__,
            error_message=error_msg,
            keywordId=keywordId,
            details={
                "content_length": content_length,
                "error": str(e)
            }
        )
        return {
            "nodes": [],
            "edges": [],
            "error": error_msg
        }

    # Save to Neo4j
    try:
        print(f"🔄 Calling saveKGToNeo4j with keywordId={keywordId}")
        print(f"   KG contains: {len(json_out.get('nodes', []))} nodes, {len(json_out.get('edges', []))} edges")
        saveKGToNeo4j(keywordId, json_out)
        print(f" saveKGToNeo4j completed without exceptions")
    except Exception as e:
        print(f" Exception caught from saveKGToNeo4j: {type(e).__name__}: {str(e)}")
        trackError(
            component="createKG->saveKGToNeo4j",
            error_type=type(e).__name__,
            error_message=str(e),
            keywordId=keywordId,
            details={
                "nodes_count": len(json_out.get("nodes", [])),
                "edges_count": len(json_out.get("edges", []))
            }
        )
        raise
    
    return json_out


def saveKGToNeo4j(keywordId: str, kg_json: dict):
    print("\n" + "=" * 80)
    print("STEP 5.*: Saving KG in Neo4j (MERGE mode - adding to existing)...")
    print("=" * 80)
    
    # Validate KG data before saving
    if not kg_json or not isinstance(kg_json, dict):
        print(" Invalid KG JSON structure")
        return
    
    nodes = kg_json.get("nodes", [])
    edges = kg_json.get("edges", [])
    
    print(f" Preparing to merge:")
    print(f"   - {len(nodes)} nodes")
    print(f"   - {len(edges)} edges")
    print(f"   - KeywordId: {keywordId}")
    
    if not nodes and not edges:
        print(" No nodes or edges to save")
        return

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            try:
                # MERGE nodes instead of CREATE (adds new or updates existing)
                nodes_created = 0
                nodes_updated = 0
                
                for i, node in enumerate(nodes):
                    label = node.get("label", "Unknown")
                    name = node.get("name", f"Node_{i}")
                    properties = node.get("properties", {})
                    properties.update({"name": name, "keywordId": keywordId})
                    
                    try:
                        # Use MERGE to create if not exists, or update if exists
                        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                        query = f"""
                            MERGE (n:{label} {{name: $name, keywordId: $keywordId}})
                            ON CREATE SET {', '.join([f'n.{k} = ${k}' for k in properties.keys()])}
                            ON MATCH SET {', '.join([f'n.{k} = ${k}' for k in properties.keys()])}
                            RETURN n
                        """
                        result = session.run(query, properties)
                        record = result.single()
                        
                        if record:
                            nodes_created += 1
                        
                    except Exception as e:
                        print(f"    Failed to merge node {i+1}: {name} - {str(e)}")
                        continue
                
                print(f" Merged {nodes_created}/{len(nodes)} nodes (created or updated)")

                # MERGE relationships instead of CREATE
                edges_created = 0
                
                for i, edge in enumerate(edges):
                    rel_type = re.sub(r"[^A-Za-z0-9_]", "_", edge.get("type", "RELATED")).upper()
                    props = edge.get("properties", {})
                    props["keywordId"] = keywordId
                    props["from"] = edge.get("from", "")
                    props["to"] = edge.get("to", "")
                    
                    if not props["from"] or not props["to"]:
                        print(f"    Skipping edge {i+1}: missing from/to nodes")
                        continue

                    try:
                        # Use MERGE to avoid duplicate relationships
                        query = f"""
                            MATCH (a {{name: $from, keywordId: $keywordId}}),
                                  (b {{name: $to, keywordId: $keywordId}})
                            MERGE (a)-[r:{rel_type} {{keywordId: $keywordId}}]->(b)
                            RETURN r
                        """
                        result = session.run(query, props)
                        record = result.single()
                        
                        if record:
                            edges_created += 1
                            
                    except Exception as e:
                        print(f"    Failed to merge edge {i+1}: {props['from']} -> {props['to']} - {str(e)}")
                        continue
                
                print(f" Merged {edges_created}/{len(edges)} edges (created or updated)")
                print(f" Successfully merged KG to Neo4j!")

            except Exception as e:
                print(f" Neo4j error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")


async def MyAgent():
    SYSTEM_PROMPT = """
    You are an intelligent agent that creates knowledge graphs from crawled web content.
    
    YOUR WORKFLOW (MUST FOLLOW IN ORDER):
    1. First, call getCrawlContent(keywordId) to fetch the crawled text data
    2. Then, call createKG(content, keywordId) to create the knowledge graph from that content, make sure this should call always and make query for createKG
    3. The createKG tool will automatically save the KG to Neo4j
    4. For all keyword create KG
    
    IMPORTANT RULES:
    - Always use BOTH tools in sequence
    - Pass the keywordId as a STRING (not ObjectId)
    - Pass the full content text to createKG
    - Report when each step is completed
    
    AVAILABLE TOOLS:
    - getCrawlContent(keywordId: str) -> Returns all crawled text for the keyword
    - createKG(content: str, keywordId: str) -> Creates KG and saves to Neo4j
    
    Example flow for keywordId "507f1f77bcf86cd799439011":
    1. Call: getCrawlContent("507f1f77bcf86cd799439011")
    2. Receive: "SLT Mobitel offers fiber internet..."
    3. Call: createKG("SLT Mobitel offers fiber internet...", "507f1f77bcf86cd799439011")
    4. Report: "Knowledge graph created and saved to Neo4j"
    """

    checkpointer = InMemorySaver()
    tools = [getCrawlContent, createKG]

    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=checkpointer
    )

    return agent

# Run Agent
async def FullAutoAgent(keywordId):
    """
    Run agent to create Knowledge Graph with error tracking
    """
    keywordId_str = str(keywordId)
    
    print("\n" + "=" * 80)
    print(f"STEP 5.1: Calling Agents for keywordId: {keywordId_str}")
    print("=" * 80)
    
    try:
        agent_executor = await MyAgent()

        print(f"🤖 Invoking agent with keywordId: {keywordId_str}")
        print(f"   Agent will: 1) Get crawl content, 2) Create KG (with auto-chunking if needed)")

        # Step 1 + 2 + 3: Crawl content → Create KG
        response = await agent_executor.ainvoke(
        {
            "messages": [
                {"role": "user", "content": f"Generate a knowledge graph for keyword ID {keywordId_str}"}
            ]
        },
        config={"configurable": {"thread_id": f"kg_{keywordId_str}"}}
        )

        print(response)
        # Check if response is valid
        if not response or "messages" not in response:
            error_msg = "Agent returned invalid response structure"
            trackError(
                component="FullAutoAgent",
                error_type="InvalidAgentResponse",
                error_message=error_msg,
                keywordId=keywordId_str,
                details={
                    "response_type": type(response).__name__,
                    "response_keys": list(response.keys()) if isinstance(response, dict) else "Not a dict"
                }
            )
            return {
                "status": "failed",
                "reason": error_msg,
                "keywordId": keywordId_str
            }
        
        # Log successful execution
        messages = response.get("messages", [])
        print(f"\n Agent completed successfully with {len(messages)} messages")
        
        return response

    except TimeoutError as e:
        trackError(
            component="FullAutoAgent",
            error_type="TimeoutError",
            error_message=f"Agent execution timed out: {str(e)}",
            keywordId=keywordId_str,
            details={"timeout_duration": "unknown"}
        )
        print(f" Agent timeout for keywordId: {keywordId_str}")
        return {
            "status": "failed",
            "reason": "Agent execution timed out",
            "keywordId": keywordId_str
        }
    
    except Exception as e:
        trackError(
            component="FullAutoAgent",
            error_type=type(e).__name__,
            error_message=str(e),
            keywordId=keywordId_str,
            details={
                "exception_type": type(e).__name__,
                "traceback": __import__('traceback').format_exc()
            }
        )
        print(f" Error in FullAutoAgent: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "reason": str(e),
            "keywordId": keywordId_str
        }


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
# async def storeRelevantUrls(keywordId):
    
#     try:
#         keywordDetails = await getKeywordById(keywordId)
        
#         keyword = keywordDetails["keyword"]

#         results = googlesearch(keyword)

#         urlList = []

#         for item in results.get("items", []):
#             print(f"Title: {item['title']}")
#             urlList.append(item['link'])
#             print(f"Link: {item['link']}\n")

#         # print(urls_list)

#         updatedValues = await keyword_collection.update_one(
#             {"_id": ObjectId(keywordId)},
#             {"$push": {"urls": {"$each": urlList}}}
#         )
#         print("Updated Values")
#         print(updatedValues)

#         if updatedValues.acknowledged:
#             print("Update successful!")
#             result = keywordId
#             return result    
#         return None
#     except Exception as e:
#         print(e)
#         return None


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

    
    # if not url_list or len(url_list) == 0:
        # print("\n" + "=" * 80)
        # print("STEP 3: Fetching Social media data from google search URLs")
        # print("=" * 80)
    #     print("Finding in here!")
    #     await storeRelevantUrls(keywordId)
    

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
    # url_list_search = updatedDetails["urls"]


    urls = [url]
    
    # if url_list_search and len(url_list_search) > 0:
    #     urls += url_list_search
    
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
    # print(resultAgent)
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