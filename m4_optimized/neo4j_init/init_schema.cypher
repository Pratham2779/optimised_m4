CREATE CONSTRAINT unique_memory_node_id IF NOT EXISTS FOR (n:MemoryNode) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT unique_memory_chain_id IF NOT EXISTS FOR (c:MemoryChain) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT unique_memory_chain_category IF NOT EXISTS FOR (c:MemoryChain) REQUIRE c.category_name IS UNIQUE;
CREATE INDEX index_memory_node_category IF NOT EXISTS FOR (n:MemoryNode) ON (n.category);
CREATE INDEX index_memory_node_session IF NOT EXISTS FOR (n:MemoryNode) ON (n.session_id);
CREATE VECTOR INDEX category_chain_embeddings IF NOT EXISTS FOR (c:MemoryChain) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX memory_node_embeddings IF NOT EXISTS FOR (n:MemoryNode) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
