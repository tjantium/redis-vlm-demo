# Redis Vector Library Demo with LlamaIndex

A demonstration of using Redis as a vector store with LlamaIndex for RAG (Retrieval Augmented Generation) applications. This project showcases how to build a question-answering system using Redis Vector Library, LlamaIndex, and open-source language models.

## Architecture
[Coming Soon] - Architecture diagram showing the interaction between Redis Vector Store, LlamaIndex, and the RAG pipeline.

## Prerequisites

- Python 3.8+
- Docker
- Git

## Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/redis-vlm-demo.git
cd redis-vlm-demo
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Redis Stack**
```bash
docker run -d --name redis \
    -p 6379:6379 \
    -p 8001:8001 \
    redis/redis-stack:latest
```

4. **Verify Redis Connection**
```bash
redis-cli ping
# Should return PONG
```

5. **Run Data Ingestion**
```bash
python ingestion.py
```

6. **Start the Agent**
```bash
python agent.py
```

## Project Structure

- `ingestion.py`: Handles document ingestion and vector store setup
- `agent.py`: Implements the RAG agent with LlamaIndex
- `data/`: Directory for your documents
- `requirements.txt`: Project dependencies

## Verifying the Setup

### Check Redis Status
```bash
# Connect to Redis CLI
redis-cli

# List all keys
KEYS *

# Get information about the vector index
FT._LIST
FT.INFO docs

# Get document count
FT.SEARCH docs "*" LIMIT 0 0
```

### Test the Agent
The agent supports both interactive mode and predefined queries. When running `agent.py`, you can:
- Use the predefined test queries
- Enter interactive mode to ask your own questions
- Type 'quit' to exit

## Components

- **Vector Store**: Redis Stack with RediSearch
- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Language Model**: TinyLlama-1.1B-Chat-v1.0
- **Framework**: LlamaIndex for RAG pipeline
- **Caching**: Semantic caching for response optimization

## Troubleshooting

### Common Issues

1. **Redis Connection Issues**
```bash
# Check if Redis container is running
docker ps | grep redis

# Restart Redis if needed
docker restart redis
```

2. **Port Conflicts**
```bash
# Check if ports are in use
lsof -i :6379
lsof -i :8001
```

3. **Memory Issues**
- Ensure you have enough RAM for the models
- Consider using smaller models if needed

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license here]

## Acknowledgments

- Redis Stack
- LlamaIndex
- HuggingFace
- TinyLlama

## Coming Soon

- [ ] Architecture Diagram
- [ ] Performance Benchmarks
- [ ] Advanced Configuration Guide
- [ ] Docker Compose Setup