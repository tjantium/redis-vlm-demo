# redis-vlm-demo
 redis vector library



## step 1

pip install -r requirements.txt


## step 2 

```bash 
Setup and Download Data
In this section, you will set up a Redis database, configure the environment, and ingest the PDF document.

Start Redis
Start a new Redis instance. You need to have Docker installed for this to work. This code uses a framework called Testcontainers, which is an open source library for providing throwaway, lightweight instances of databases, message brokers, web browsers, or just about anything that can run in a Docker container.

💡 Note that you will need to explicitly stop the container created for Redis later on in this notebook. There is a section called Stop Redis, which will help you with this. Failing to do this will leave a container running blocking the port 6379.

```



# Connect to Redis CLI
redis-cli

# List all keys
KEYS *

# Get information about the vector index
FT._LIST
FT.INFO docs

# Get document count
FT.SEARCH docs "*" LIMIT 0 0