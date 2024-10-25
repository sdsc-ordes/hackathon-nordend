import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import json
import os
import pandas as pd
from openai import OpenAI
from code_editor import code_editor

# Setting the API key




# Function to get the assistant's response
def get_assistant_response(messages):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model="gpt-4o",
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
def get_statistics(tx):
    result = tx.run("""// Total row
                    MATCH (n)
                    RETURN 'Total' AS Label, count(n) AS NodeCount
                    UNION
                    // Node count per label
                    MATCH (n)
                    RETURN labels(n) AS Label, count(n) AS NodeCount
                    ORDER BY NodeCount DESC
                    """).to_df()
    return result 
def get_node_labels(tx):
    result = tx.run("MATCH (n) RETURN DISTINCT labels(n) AS labels")
    return list(set([record["labels"][0] for record in result if record["labels"]]))

def get_property_counts(tx,label):
    result = tx.run(f"""CALL db.schema.nodeTypeProperties() YIELD propertyName
        WITH propertyName
        MATCH (n:`{label}`)
        WHERE n[propertyName] IS NOT NULL
        RETURN propertyName, COUNT(DISTINCT n[propertyName]) AS uniqueCount
        ORDER BY propertyName""").to_df()
    result['Label'] = label
    return result


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

def get_top_n_nodes_counts(tx, limit):
    query = f"""
        MATCH (n)
        RETURN labels(n) AS Name, count(n) AS Count
        ORDER BY Count DESC
        LIMIT {limit};
    """
    return tx.run(query).to_df()

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

def run_cypher_query(query):
    try:
        with driver.session() as session:
            result = session.run(query)
            # Collect all records into a list
            records = [record.data() for record in result]
            return records
    except Exception as e:
        # Return None in case of an error
        return None
    
def explain_cypher_query(query):
    try:
        with driver.session() as session:
            result = session.run("EXPLAIN " + query)
            # Collect all records into a list
            records = result
            return records
    except Exception as e:
        # Return None in case of an error
        return None

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

        
        with driver.session() as session:
            # Get node labels and relationships
            st.success("Connection successfull!")

            st.session_state["connected"] = True
            node_count = session.execute_read(get_node_count)
            relationship_count = session.execute_read(get_relationship_count)
            statistics = session.execute_read(get_statistics)
            node_labels = list(set(session.execute_read(get_node_labels)))
            #Display cards with the graph statistics
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Nodes", node_count)
            col2.metric("Number of Relationships", relationship_count)
            col3.metric("Entity Types", len(node_labels))

            limit = 10
            st.subheader(f"The {limit} nodes with most occurrences are:")
            st.write(session.execute_read(get_top_n_nodes_counts, limit))

            # List entity (node) names with their properties
            st.subheader("The properties of each node are:")
            node_properties_labels = []
            label_dict = {}
            for label in node_labels:
                node_properties_labels = list(set(session.execute_read(get_all_node_properties_with_labels,label)))
                label_dict[label] = node_properties_labels
            st.json(label_dict, expanded=False)

            label_json = json.dumps(label_dict)
            def update_dict_with_descriptions(main_dict, description_list):
                for item in description_list:
                    label = item['label']
                    description = item['description']
                    properties = item['properties']
                    
                    # Check if the label exists in the main dictionary
                    if label in main_dict:
                        # Add the description
                        main_dict[label]['description'] = description
            
                        # Append the property descriptions to the 'properties' list
                        for i, prop_desc in enumerate(properties):
                            for i,element in enumerate(main_dict[label]['properties']):
                                if element == prop_desc['property_name']:
                                    main_dict[label]['properties'][i] = main_dict[label]['properties'][i] + "\n" + prop_desc['property_description']
                return main_dict
            relationships = session.execute_read(get_relationship_types)

            #st.subheader("Relationship names")
            #st.write(", ".join(relationships))

            node_labels = sorted(node_labels)
            node_properties_labels = []
            relationships = session.execute_read(get_relationship_types)

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

            label_dict = {}
            all_labels_df = pd.DataFrame()
            with st.spinner('Distilling the graph...'):
                for label in node_labels:
                    #label_df = session.execute_read(get_property_counts,label)
                    #all_labels_df = pd.concat([all_labels_df,label_df])
                    node_properties_labels = list(set(session.execute_read(get_all_node_properties_with_labels,label)))
                    label_dict[label] = {}
                    label_dict[label]['properties'] = sorted(node_properties_labels)
                    label_dict[label]['outgoing_relations'] = sorted([item for item in all_relationships if item.startswith(label)])
                    label_dict[label]['ingoing_relations'] = sorted([item for item in all_relationships if item.endswith(label) if not item.startswith(label)])
            def open_jsonl_file(file_path):
                with open(file_path, 'r') as f:
                    json_objects = []
                    for line in f:
                        json_objects.append(json.loads(line))  # Parse each line into a JSON object
                return json_objects
            label_json_desc = open_jsonl_file('schema_description.jsonl')

            update_dict_with_descriptions(label_dict, label_json_desc)
            with st.expander("See full relationships list:", expanded=False):
                st.write(label_dict)

            label_json = json.dumps(label_dict)


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
                   For each node label, use only the following schema:
                   {label_json}

                   Pay attention to the descriptions. For example, do not search for an element in english if the field only contains elements in german.
                   """]
        st.session_state['llm_messages'] = [{"role": "system", "content": prompts[0]},]

        def process_query(use_llm=True):

            if use_llm:
                key_suffix = "_llm"
                query_label = "Insert your query:"
                button_label = "Run default query"
            else:
                key_suffix = ""
                query_label = "Insert your Cypher query:"
                button_label = "Run your Cypher query"
            user_input_key = "user_input"+key_suffix
            output_key = "output"+key_suffix
            def process_user_input():
                user_input = st.session_state[user_input_key]
                st.session_state['llm_messages'].append({"role": "user", "content": user_input + f"Write only the query with no formatting, use only the schema available here: \n{label_json}"})
                output = get_assistant_response(st.session_state['llm_messages']).strip("`")
                st.session_state[output_key] = output

            if use_llm:
                st.text_area(label=query_label, placeholder="Enter text here", key=user_input_key, on_change=process_user_input)
            else:
                st.session_state[output_key] = None


            if output_key in st.session_state:
                if not use_llm:
                    st.text(query_label)
                response_dict = code_editor(st.session_state[output_key], lang="sql",
                                            key='code'+key_suffix, allow_reset=True)
                if len(response_dict['text']) > 0:
                    st.session_state[output_key] = response_dict['text']

                # Button to run the query
                if st.button(button_label):
                    # Save clicked status in the session state
                    st.session_state['query_clicked'] = True

                    # Check if the button was clicked
                    if st.session_state.get('query_clicked'):
                        # Run the query and get the result
                        try:
                            result = run_cypher_query(st.session_state[output_key])
                        except Exception as e:
                            raise e
                            # Handle different cases based on the query result
                        if result is None:
                            st.error(
                                "Oh no, an error occurred while running the query.")
                        elif len(result) == 0:
                            st.warning("Oh no, empty list!")
                        else:
                            st.write(result)


        with st.spinner('Asking ChatGPT...'):
            process_query(use_llm=True)

        # User defined Cypher query
        process_query(use_llm=False)


        driver.close()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        raise e
