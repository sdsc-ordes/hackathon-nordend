# De-cypher
## Escape the hell of cypher using the power of LLM's

De-cypher is the result of a 2 day Open Research Data Hackathon organized by EPFL Open Science office and Swiss Data Science Center. It's main goal is to increase accessibility to data by non-query proficient users by providing a user interface to ask natural language questions over a Neo4J property graph database.
Although the data chosen to implement this use case is the [DemocraSci](https://zenodo.org/records/13920293) knowledge graph, its structure is modular, and can be plugged into any Neo4J endpoint. Users can validate their database has loaded correctly with the help of some statistics, and then proceed to generate Cypher queries that can be ran from within the Streamlit front-end.
Each time a query is ran, it will be validated for correctness. In case the LLM has halucinated some properties/logic, the errors returned will guide the user into a productive iterative dialogue with the LLM in order to improve the query.

## Installation instructions

In order to run de-cypher, you need the following:
- A machine with [python installed ](https://realpython.com/installing-python/) and [pip](https://pypi.org/project/pip/) installed.
- A Neo4J property graph, exposed to your local network. By default, when initializing a DB in Neo4j, it will be exposed on [Bolt port](https://neo4j.com/docs/operations-manual/current/configuration/connectors/) bolt://localhost:7687. Make sure to remember the username and password set when initializing the DB, as you will need this later in the front-end to access the DB.
- A question for the data!

1. enter your python environment
2. `pip install -r requirements.txt`
3. Initialize your neo4j database, make sure to import the neo4j.dump file
4. start the database
5. run `python streamlit run app.py`
6. login on the frontend (check the terminal for the port to use in your case, usually localhost:8501)
7. login on the front end using your neo4j username and password which you created when initializing the database
8. validate your graph loaded correctly by checking the statistics
9. enter a natural language question
10. wait for the model to return a query and run it!

## Features
- Neo4J connector
- Standard statistical queries for some data metrics and extracting the schema (relationships & properties) that can be used to query
- Iterative LLM powered query window
