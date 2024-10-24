import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import json
import os
from openai import OpenAI

# Setting the API key




# Function to get the assistant's response
def get_assistant_response(messages):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

 
# Function to create a Neo4j driver instance
def create_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))
def get_all_node_properties_with_labels(tx,label):
    query = f"""
        MATCH (n:`{label}`)
        WITH DISTINCT keys(n) AS propertyKeys
        UNWIND propertyKeys AS key
        RETURN DISTINCT key AS properties
        """
    result = tx.run(query)
    return list(set([record["properties"] for record in result if (record['properties'])]))

def get_node_labels(tx):
    result = tx.run("MATCH (n) RETURN DISTINCT labels(n) AS labels")
    return list(set([record["labels"][0] for record in result if record["labels"]]))

# Function to query relationships between node labels
def get_relationship_types(tx):
    result = tx.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship")
    return [record["relationship"] for record in result]

# Function to query the number of nodes in the graph
def get_node_count(tx):
    result = tx.run("MATCH (n) RETURN count(n) AS count")
    return result.single()["count"]

# Function to query the number of relationships in the graph
def get_relationship_count(tx):
    result = tx.run("MATCH ()-[r]->() RETURN count(r) AS count")
    return result.single()["count"]


# Function to query node labels
def get_node_labels(tx):
    result = tx.run("MATCH (n) RETURN DISTINCT labels(n) AS labels")
    return [record["labels"][0] for record in result if record["labels"]]

# Function to query relationships between node labels

def get_properties(tx):
    result = tx.run("""CALL db.schema.nodeTypeProperties()
        YIELD nodeLabels, propertyName
        WITH nodeLabels[0] AS label, COLLECT(propertyName) AS properties
        RETURN {
        type: 'entity',
        label: label,
        properties: properties
        } AS schema
        ORDER BY label""")
    return list(set([record["label"] + "_" + "_".join(record['properties']) for record in result if (record['label'] and record['properties'])]))
def get_relationship_types(tx):
    result = tx.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship")
    return list(set([record["relationship"] for record in result]))

# Function to get source and target labels for a specific relationship type
def get_relationship_pairs(tx, relationship_type):
    query = f"""
    MATCH (n)-[r:{relationship_type}]->(m) 
    RETURN DISTINCT labels(n)[0] AS start_label, labels(m)[0] AS end_label
    """
    result = tx.run(query)
    return list(set([(record["start_label"], record["end_label"]) for record in result]))
# Streamlit app title
st.title("Neo4j Database Schema Visualization (with Neo4j Driver)")

# Neo4j connection details
st.sidebar.header("Neo4j Connection")
uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
username = st.sidebar.text_input("Username", "neo4j")
password = st.sidebar.text_input("Password", type="password")

# Button to connect to the database
if 'connected' in st.session_state or st.sidebar.button("Connect"):
    try:
        # Establish connection to Neo4j using the official driver
        driver = create_driver(uri, username, password)
        st.success("Connection successfull!")

        st.session_state["connected"] = True
        
        with driver.session() as session:
            # Get node labels and relationships

            node_count = session.execute_read(get_node_count)
            
            relationship_count = session.execute_read(get_relationship_count)
            node_labels = list(set(session.execute_read(get_node_labels)))
            node_properties_labels = []
            label_dict = {}
            for label in node_labels:
                node_properties_labels = list(set(session.execute_read(get_all_node_properties_with_labels,label)))
                label_dict[label] = node_properties_labels
            st.write(label_dict)
            label_json = json.dumps(label_dict)
            relationships = session.execute_read(get_relationship_types)

            #Display cards with the graph statistics
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Nodes", node_count)
            col2.metric("Number of Relationships", relationship_count)
            col3.metric("Entity Types", len(node_labels))

            # List entity (node) names
            st.subheader("Entity Names")

            #st.subheader("Relationship names")
            #st.write(", ".join(relationships))

            #Create a NetworkX graph for schema
            G = nx.DiGraph()

            # Add nodes (labels as node types)
            for label in node_labels:
                G.add_node(label, color='blue', size=1000)

            # Add edges (relationships as directed edges between node types)
            all_relationships = []
            for rel in relationships:
                relationship_pairs = session.execute_read(get_relationship_pairs, rel)
                for start_label, end_label in relationship_pairs:
                    all_relationships.append(f"{start_label} --[:{rel}]--> {end_label}")

                    G.add_edge(start_label, end_label, label=rel)
            st.subheader("Relationship list")
            st.write(all_relationships)
        #Plot the graph using NetworkX and matplotlib
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        def visualize_graph(G):
                from random import choice
                import matplotlib.colors as mcolors
                # Initialize the PyVis Network object
                net = Network(notebook=True, height="600px", width="100%", directed=True)

                # Map of colors to be used for relationships
                edge_colors = list(mcolors.CSS4_COLORS.values())

                # Add nodes with labels and smaller sizes
                for node in G.nodes():
                    net.add_node(node, label=node, size=20, color='lightblue')

                # Add edges with different colors based on relationship type
                for u, v, data in G.edges(data=True):
                    rel_type = data['label']
                    edge_color = choice(edge_colors)  # Pick a random color for each relationship
                    net.add_edge(u, v, label=rel_type, color=edge_color)

                try:
                    path = '/tmp'
                    net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                # Save and read graph as HTML file (locally)
                except:
                    path = '/html_files'
                    net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

                components.html(HtmlFile.read(), height=435)
        #Call the function to visualize
        #visualize_graph(G)
        prompts = [f"""Generate a neo4j cypher query corresponding to the natural language question given by the user. Only return formatted Cypher queries. Keep in mind you can only use the following nodes, edges, relationships and properties:
                   For each node label, use only the following properties:
                   {label_json}
                   Only use the following relationships:
                   {relationships}
                   """
                   ]
        st.session_state['llm_messages'] = [{"role": "system", "content": prompts[0]},]

        def process_user_input():
            template_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship"
            user_input = st.session_state["user_input"]
            st.session_state['llm_messages'].append({"role": "user", "content" : user_input})
            output = get_assistant_response(st.session_state['llm_messages'])
            st.session_state["output"] = template_query + "\n" + output
        with st.spinner('Wait for it...'):
            st.text_area(label="User Input", placeholder="Enter stuff here", key="user_input", on_change=process_user_input)
            if "output" in st.session_state:
                st.code(st.session_state["output"],language="cypher")
        driver.close()
    
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
