#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import argparse
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import openai
from chromadb.config import Settings

from llm_client import get_openai_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MISSION_NAMES = {
    'apollo_11': 'Apollo 11',
    'apollo_13': 'Apollo 13',
    'challenger': 'Challenger',
}

# Preferred chunk break points, best first: end of sentence, end of line, any whitespace.
BREAK_PATTERNS = [
    re.compile(r'[.!?]["\')\]]?\s'),
    re.compile(r'\n'),
    re.compile(r'\s'),
]

MAX_RETRIES = 5
RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError,
                    openai.APITimeoutError, openai.InternalServerError)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int]]:
    """
    Split text into (start, end) character spans.

    Rules:
    - Every span is at most chunk_size characters long.
    - Each span starts exactly chunk_overlap characters before the previous span ended.
    - A span ends at a sentence end, line break, or space when one exists in its
      second half, so chunks rarely cut a sentence in the middle.
    """
    if len(text) <= chunk_size:
        return [(0, len(text))]

    spans = []
    start = 0
    while True:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_break_point(text, start, end, chunk_size, chunk_overlap)
        spans.append((start, end))
        if end >= len(text):
            return spans
        start = end - chunk_overlap


def find_break_point(text: str, start: int, end: int, chunk_size: int, chunk_overlap: int) -> int:
    """Return the best chunk end in the window (min_end, end]."""
    # min_end > start + chunk_overlap guarantees the next chunk starts after this one.
    min_end = start + max(chunk_size // 2, chunk_overlap + 1)
    for pattern in BREAK_PATTERNS:
        matches = list(pattern.finditer(text, min_end, end))
        if matches:
            return matches[-1].end()
    return end


def clean_text(text: str) -> str:
    """Collapse repeated spaces and blank lines left over from OCR."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""

    def __init__(self,
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline

        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        self.openai_client = get_openai_client(openai_api_key)

        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Embeddings are computed here with OpenAI, so ChromaDB needs no embedding function.
        # The model name is stored so rag_client.py embeds questions with the same model.
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                'embedding_model': embedding_model,
                'description': 'NASA Apollo 11, Apollo 13 and Challenger mission documents',
            },
            configuration={'hnsw': {'space': 'cosine'}},
        )

        stored_model = (self.collection.metadata or {}).get('embedding_model')
        if stored_model and stored_model != embedding_model:
            raise ValueError(
                f"Collection '{collection_name}' was built with '{stored_model}', not "
                f"'{embedding_model}'. Use the same model or a new --collection-name."
            )

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata

        Args:
            text: Text to chunk
            metadata: Base metadata for the text

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        if not text.strip():
            return []

        spans = split_text(text, self.chunk_size, self.chunk_overlap)
        chunks = []
        for index, (start, end) in enumerate(spans):
            chunk_metadata = {
                **metadata,
                'chunk_index': index,
                'total_chunks': len(spans),
                'chunk_start': start,
                'chunk_length': end - start,
                'chunk_size_setting': self.chunk_size,
                'chunk_overlap_setting': self.chunk_overlap,
            }
            chunks.append((text[start:end], chunk_metadata))
        return chunks

    def existing_ids(self, doc_ids: List[str]) -> set:
        """Return the subset of doc_ids that already exist in the collection."""
        if not doc_ids:
            return set()
        return set(self.collection.get(ids=doc_ids, include=[])['ids'])

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        return doc_id in self.existing_ids([doc_id])

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection

        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)

            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)

        Args:
            source_pattern: Pattern to match source names

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get(include=['metadatas'])

            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            result = self.collection.get(
                where={'$and': [{'source': source}, {'mission': mission}]},
                include=[]
            )
            return result['ids']

        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get OpenAI embeddings for many texts in one API call, retrying on temporary errors

        Args:
            texts: Texts to embed

        Returns:
            One embedding vector per text, in the same order
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.openai_client.embeddings.create(model=self.embedding_model, input=texts)
                return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            except RETRYABLE_ERRORS as e:
                if attempt == MAX_RETRIES:
                    raise
                wait_seconds = 2 ** attempt
                logger.warning(f"Embedding request failed ({e.__class__.__name__}), "
                               f"retry {attempt}/{MAX_RETRIES - 1} in {wait_seconds}s")
                time.sleep(wait_seconds)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.get_embeddings([text])[0]

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        mission = metadata.get('mission', self.extract_mission_from_path(file_path))
        source = metadata.get('source', file_path.stem)
        return f"{mission}_{source}_chunk_{metadata['chunk_index']:04d}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction

        Args:
            file_path: Path to text file

        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = clean_text(f.read())

            if not content:
                return []

            mission = self.extract_mission_from_path(file_path)

            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': file_path.as_posix(),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': mission,
                'mission_name': MISSION_NAMES.get(mission, 'Unknown'),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()

        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'

        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'

        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'

        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)

        Args:
            base_path: Base directory path

        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []

        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")

                # Find only text files
                text_files = sorted(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)

        logger.info(f"Total text files to process: {len(filtered_files)}")

        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1

        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered_files

    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]],
                                    file_path: Path, batch_size: int = 50,
                                    update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling

        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add

        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        if update_mode not in ('skip', 'update', 'replace'):
            raise ValueError(f"Unknown update_mode: {update_mode}")

        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        doc_ids = [self.generate_document_id(file_path, metadata) for _, metadata in documents]

        if update_mode == 'replace':
            old_ids = self.get_file_documents(file_path)
            if old_ids:
                self.collection.delete(ids=old_ids)
                logger.info(f"  Replace mode: deleted {len(old_ids)} old chunks")

        for batch_start in range(0, len(documents), batch_size):
            batch_ids = doc_ids[batch_start:batch_start + batch_size]
            batch_docs = documents[batch_start:batch_start + batch_size]
            existing = self.existing_ids(batch_ids)

            new_rows = [(doc_id, text, meta) for doc_id, (text, meta) in zip(batch_ids, batch_docs)
                        if doc_id not in existing]
            existing_rows = [(doc_id, text, meta) for doc_id, (text, meta) in zip(batch_ids, batch_docs)
                             if doc_id in existing]

            if update_mode == 'skip':
                stats['skipped'] += len(existing_rows)
                existing_rows = []

            rows_to_embed = new_rows + existing_rows
            if not rows_to_embed:
                continue

            embeddings = self.get_embeddings([text for _, text, _ in rows_to_embed])
            new_embeddings = embeddings[:len(new_rows)]
            existing_embeddings = embeddings[len(new_rows):]

            if new_rows:
                self.collection.add(
                    ids=[row[0] for row in new_rows],
                    documents=[row[1] for row in new_rows],
                    metadatas=[row[2] for row in new_rows],
                    embeddings=new_embeddings,
                )
                stats['added'] += len(new_rows)

            if existing_rows:
                self.collection.update(
                    ids=[row[0] for row in existing_rows],
                    documents=[row[1] for row in existing_rows],
                    metadatas=[row[2] for row in existing_rows],
                    embeddings=existing_embeddings,
                )
                stats['updated'] += len(existing_rows)

        # In update mode, remove old chunks that no longer exist (for example after
        # a larger chunk_size produced fewer chunks for this file).
        if update_mode == 'update':
            stale_ids = set(self.get_file_documents(file_path)) - set(doc_ids)
            if stale_ids:
                self.collection.delete(ids=list(stale_ids))
                logger.info(f"  Update mode: deleted {len(stale_ids)} stale chunks")

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip',
                              batch_size: int = 50) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB

        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            batch_size: Number of chunks embedded per API call

        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path)
        if not files:
            logger.warning(f"No text files found under {base_path}")
            return stats

        for file_number, file_path in enumerate(files, start=1):
            mission = self.extract_mission_from_path(file_path)
            mission_stats = stats['missions'].setdefault(
                mission, {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0})

            try:
                documents = self.process_text_file(file_path)
                if not documents:
                    logger.warning(f"[{file_number}/{len(files)}] {file_path.name}: no text, skipped")
                    continue

                logger.info(f"[{file_number}/{len(files)}] {file_path.name}: {len(documents)} chunks")
                result = self.add_documents_to_collection(documents, file_path, batch_size, update_mode)

                stats['files_processed'] += 1
                stats['total_chunks'] += len(documents)
                stats['documents_added'] += result['added']
                stats['documents_updated'] += result['updated']
                stats['documents_skipped'] += result['skipped']
                mission_stats['files'] += 1
                mission_stats['chunks'] += len(documents)
                for key in ('added', 'updated', 'skipped'):
                    mission_stats[key] += result[key]

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Failed to process {file_path}: {e}")

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        return {
            'collection_name': self.collection.name,
            'document_count': self.collection.count(),
            'metadata': self.collection.metadata,
            'chroma_dir': self.chroma_persist_directory,
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing

        Args:
            query_text: Query text
            n_results: Number of results to return

        Returns:
            Query results
        """
        return self.collection.query(
            query_embeddings=[self.get_embedding(query_text)],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get(include=['metadatas'])

            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {},
                'sources': {},
            }
            chunk_lengths = []

            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                source = metadata.get('source', 'unknown')

                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1

                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1

                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1

                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1

                # Count by source file
                stats['sources'][source] = stats['sources'].get(source, 0) + 1

                if 'chunk_length' in metadata:
                    chunk_lengths.append(metadata['chunk_length'])

            if chunk_lengths:
                stats['average_chunk_length'] = round(sum(chunk_lengths) / len(chunk_lengths), 1)
                stats['max_chunk_length'] = max(chunk_lengths)

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def print_collection_stats(pipeline: ChromaEmbeddingPipelineTextOnly) -> None:
    """Log collection size and per-mission, per-category and per-source chunk counts."""
    stats = pipeline.get_collection_stats()
    logger.info("=" * 60)
    logger.info(f"COLLECTION STATISTICS: {pipeline.collection_name} ({pipeline.chroma_persist_directory})")
    logger.info("=" * 60)

    if 'error' in stats:
        logger.info(f"Collection is empty or unreadable: {stats['error']}")
        return

    logger.info(f"Total chunks: {stats['total_documents']}")
    if 'average_chunk_length' in stats:
        logger.info(f"Average chunk length: {stats['average_chunk_length']} chars "
                    f"(max {stats['max_chunk_length']})")

    for title, key in [("Chunks per mission", 'missions'),
                       ("Chunks per document category", 'document_categories'),
                       ("Chunks per data type", 'data_types'),
                       ("Chunks per source file", 'sources')]:
        logger.info(f"{title}:")
        for name, count in sorted(stats[key].items(), key=lambda item: -item[1]):
            logger.info(f"  {name}: {count}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='./data_text', help='Path to data directories')
    parser.add_argument('--openai-key', default=None,
                        help='OpenAI API key (default: OPENAI_API_KEY from .env)')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')

    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error('--chunk-size must be at least 1')
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error('--chunk-overlap must be >= 0 and smaller than --chunk-size')
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if not (args.openai_key or os.getenv('OPENAI_API_KEY')):
        parser.error('No API key. Set OPENAI_API_KEY in .env or pass --openai-key')

    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return

    # If stats only, show collection statistics and exit
    if args.stats_only:
        print_collection_stats(pipeline)
        return

    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode} "
                f"(chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap})")
    start_time = time.time()

    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode,
                                           batch_size=args.batch_size)

    end_time = time.time()
    processing_time = end_time - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    # Mission breakdown
    logger.info("Mission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")

    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"Collection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")

    # Test query if provided
    if args.test_query:
        logger.info(f"Testing query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
