import requests
import arxiv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import hashlib
import logging
from typing import List, Dict, Optional
import io

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document fetching, processing, and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize document processor with chunking parameters"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}")
    
    def fetch_arxiv_paper(self, arxiv_id: str) -> Dict:
        """
        Fetch paper from arXiv
        
        Args:
            arxiv_id (str): arXiv paper ID (e.g., '2301.07041')
            
        Returns:
            Dict: Paper metadata and content
        """
        try:
            logger.info(f"Fetching arXiv paper: {arxiv_id}")
            
            # Clean arxiv_id (remove arxiv: prefix if present)
            clean_id = arxiv_id.replace("arxiv:", "").strip()
            
            # Search for the paper
            search = arxiv.Search(id_list=[clean_id])
            paper = next(search.results())
            
            # Download PDF
            logger.info(f"Downloading PDF for {clean_id}")
            pdf_url = paper.pdf_url
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            paper_data = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'published': paper.published.isoformat() if paper.published else None,
                'pdf_content': response.content,
                'source_id': clean_id,
                'source_type': 'arxiv',
                'url': paper.entry_id
            }
            
            logger.info(f"Successfully fetched arXiv paper: {paper.title}")
            return paper_data
            
        except StopIteration:
            raise ValueError(f"arXiv paper with ID '{arxiv_id}' not found")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching arXiv paper {arxiv_id}: {str(e)}")
            raise Exception(f"Error fetching arXiv paper: {str(e)}")
    
    def fetch_doi_paper(self, doi: str) -> Dict:
        """
        Fetch paper metadata from DOI
        
        Args:
            doi (str): DOI of the paper
            
        Returns:
            Dict: Paper metadata and content
        """
        try:
            logger.info(f"Fetching DOI paper: {doi}")
            
            # Get metadata from Crossref
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()['message']
            
            # Extract basic metadata
            title = data.get('title', ['Unknown Title'])[0]
            authors = []
            for author in data.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                full_name = f"{given} {family}".strip()
                if full_name:
                    authors.append(full_name)
            
            # Try to get PDF URL (this may not always work)
            pdf_url = None
            if 'link' in data:
                for link in data['link']:
                    if 'application/pdf' in link.get('content-type', ''):
                        pdf_url = link['URL']
                        break
            
            # Try alternative PDF sources
            if not pdf_url and 'URL' in data:
                # Some publishers provide direct PDF links
                base_url = data['URL']
                potential_pdf_urls = [
                    f"{base_url}.pdf",
                    f"{base_url}/pdf",
                    base_url.replace('doi.org', 'sci-hub.se')  # Note: Use responsibly
                ]
                
                for test_url in potential_pdf_urls:
                    try:
                        test_response = requests.head(test_url, timeout=10)
                        if test_response.status_code == 200:
                            pdf_url = test_url
                            break
                    except:
                        continue
            
            if not pdf_url:
                raise Exception("PDF not available for this DOI. Try using the arXiv version if available.")
            
            # Download PDF
            logger.info(f"Downloading PDF from: {pdf_url}")
            pdf_response = requests.get(pdf_url, timeout=30)
            pdf_response.raise_for_status()
            
            paper_data = {
                'title': title,
                'authors': authors,
                'abstract': data.get('abstract', ''),
                'published': data.get('published-print', {}).get('date-parts', [[None]])[0],
                'pdf_content': pdf_response.content,
                'source_id': doi,
                'source_type': 'doi',
                'url': data.get('URL', '')
            }
            
            logger.info(f"Successfully fetched DOI paper: {title}")
            return paper_data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching DOI paper: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching DOI paper {doi}: {str(e)}")
            raise Exception(f"Error fetching DOI paper: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF bytes
        
        Args:
            pdf_content (bytes): PDF file content
            
        Returns:
            str: Extracted text
        """
        try:
            logger.info("Extracting text from PDF")
            
            # Create a BytesIO object from the PDF content
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i}: {str(e)}")
                    continue
            
            if not text.strip():
                raise Exception("No text could be extracted from the PDF")
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split document into chunks with metadata
        
        Args:
            text (str): Document text
            metadata (Dict): Document metadata
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        try:
            logger.info(f"Chunking document: {metadata.get('title', 'Unknown')}")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            chunked_docs = []
            for i, chunk in enumerate(chunks):
                # Create unique chunk ID
                chunk_id = hashlib.md5(
                    f"{metadata['source_id']}_{i}".encode()
                ).hexdigest()
                
                chunk_data = {
                    'chunk_id': chunk_id,
                    'content': chunk.strip(),
                    'chunk_index': i,
                    'source_id': metadata['source_id'],
                    'title': metadata['title'],
                    'authors': metadata['authors'],
                    'source_type': metadata['source_type'],
                    'url': metadata.get('url', ''),
                    'abstract': metadata.get('abstract', '')
                }
                
                # Only add non-empty chunks
                if chunk_data['content']:
                    chunked_docs.append(chunk_data)
            
            logger.info(f"Created {len(chunked_docs)} chunks from document")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise Exception(f"Error chunking document: {str(e)}")
    
    def process_document(self, source_id: str, source_type: str = 'arxiv') -> List[Dict]:
        """
        Complete document processing pipeline
        
        Args:
            source_id (str): Document ID (arXiv ID or DOI)
            source_type (str): 'arxiv' or 'doi'
            
        Returns:
            List[Dict]: Processed chunks
        """
        try:
            # Fetch document
            if source_type.lower() == 'arxiv':
                doc_data = self.fetch_arxiv_paper(source_id)
            elif source_type.lower() == 'doi':
                doc_data = self.fetch_doi_paper(source_id)
            else:
                raise ValueError("Source type must be 'arxiv' or 'doi'")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(doc_data['pdf_content'])
            
            # Chunk document
            chunks = self.chunk_document(text, doc_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {source_id}: {str(e)}")
            raise
