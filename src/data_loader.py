
import os
import pandas as pd
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Data Structure: All files directly in data/ folder
# Files:
# 1. ai-schedule-change-policy-updated.pdf
# 2. air-india-coc.pdf
# 3. air-india-general-booking-policies-oct2025.pdf
# 4. U.S. Department of Transportation - Air Consumer Privacy.pdf
# 5. U.S. Department of Transportation - Aircraft Dissinection.pdf
# 6. U.S. Department of Transportation - Aviation Industry Bankruptcy and Service Cessation.pdf
# 7. U.S. Department of Transportation - Implementation of the Consumer Credit Protection Act With Respect to Air Carriers and Foreign Air Carriers.pdf
# 8. U.S. Department of Transportation - Refunds and Other Consumer Protections.pdf
# 9. U.S. Department of Transportation - Refunds for Airline Fare and Ancillary Service Fees.pdf

DATA_DIR = Path("data")

class TravelDataLoader:
    """Loads travel documents (PDFs) directly from data/ folder"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_all_pdfs(self) -> List[Document]:
        """Load all PDFs directly from data/ folder (no subfolders)"""
        documents = []
        
        if not DATA_DIR.exists():
            print(f"Error: Directory {DATA_DIR} does not exist")
            return documents
        
        # Get all PDF files directly from data/ folder
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        print(f"\n📂 Found {len(pdf_files)} PDF files in data/")
        print("=" * 60)
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Categorize based on filename
                category = self._categorize_document(pdf_file.name)
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": str(pdf_file.name),
                        "category": category,
                        "file_type": "pdf",
                        "full_path": str(pdf_file)
                    })
                
                documents.extend(docs)
                print(f"  ✓ Loaded: {pdf_file.name} ({len(docs)} pages) - Category: {category}")
                
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")
        
        print("=" * 60)
        print(f"✅ Total documents loaded: {len(documents)}")
        
        return documents
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename"""
        filename_lower = filename.lower()
        
        if "air-india" in filename_lower or "ai-schedule" in filename_lower:
            return "air_india_policies"
        elif "u.s. department" in filename_lower or "transportation" in filename_lower:
            return "us_dot_regulations"
        elif "booking" in filename_lower or "policy" in filename_lower:
            return "booking_policies"
        elif "refund" in filename_lower:
            return "refund_policies"
        elif "privacy" in filename_lower:
            return "privacy_policies"
        else:
            return "general"
    
    def load_csvs(self) -> List[Document]:
        """Load CSV files if any exist in data/ folder"""
        documents = []
        csv_files = list(DATA_DIR.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in data/ folder")
            return documents
        
        print(f"\n📊 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Convert each row to a document
                for idx, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(csv_file.name),
                            "category": "routes",
                            "file_type": "csv",
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
                
                print(f"  ✓ Loaded: {csv_file.name} ({len(df)} rows)")
                
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
        
        return documents
    
    def load_all_travel_documents(self) -> List[Document]:
        """Load all travel documents (PDFs and CSVs) from data/ folder"""
        print("\n📂 Loading travel knowledge base from data/ folder...")
        
        # Load PDFs
        pdf_docs = self.load_all_pdfs()
        
        # Load CSVs (if any)
        csv_docs = self.load_csvs()
        
        all_documents = pdf_docs + csv_docs
        
        print(f"\n📊 Total documents loaded: {len(all_documents)}")
        print(f"   - PDFs: {len(pdf_docs)}")
        print(f"   - CSVs: {len(csv_docs)}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            print("No documents to split")
            return []
        
        print(f"\n✂️  Splitting {len(documents)} documents into chunks...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        if chunks:
            print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
        
        return chunks
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about loaded documents"""
        if not documents:
            return {"total_documents": 0, "by_category": {}, "total_chars": 0}
        
        stats = {
            "total_documents": len(documents),
            "by_category": {},
            "by_source": {},
            "total_chars": sum(len(doc.page_content) for doc in documents)
        }
        
        for doc in documents:
            category = doc.metadata.get("category", "unknown")
            source = doc.metadata.get("source", "unknown")
            
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        return stats


if __name__ == "__main__":
    loader = TravelDataLoader()
    docs = loader.load_all_travel_documents()
    chunks = loader.split_documents(docs)
    stats = loader.get_document_stats(chunks)
    
    print("\n📊 Document Statistics:")
    print(f"   Total chunks: {stats['total_documents']}")
    print(f"   By category: {stats['by_category']}")
    print(f"   Total characters: {stats['total_chars']:,}")