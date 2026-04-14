from fastapi import APIRouter, HTTPException
from pathlib import Path

from config.setting import get_settings
from src.ingestion.markdown_processor import MarkdownProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create Router for Graph API
router = APIRouter(prefix="/api/graph", tags=["Graph"])
settings = get_settings()

DATA_DIR = Path(settings.paths.data_dir).resolve()
processor = MarkdownProcessor()

@router.get("/")
def get_graph_data():
    """
    Generate Nodes and Edges for the Obsidian-style Graph View.
    It scans all Markdown files and extracts bidirectional links [[Link]].
    """
    nodes = []
    edges = []
    existing_files = set()
    
    try:
        # Step 1: Find all existing files (Nodes)
        for file_path in DATA_DIR.glob("**/*.md"):
            node_id = file_path.stem  # filename without .md
            existing_files.add(node_id)
            nodes.append({
                "id": node_id,
                "label": node_id,
                "group": "existing" # To colorize real notes vs ghost notes in UI
            })
            
        # Step 2: Parse content to build connections (Edges)
        ghost_nodes = set()
        
        for file_path in DATA_DIR.glob("**/*.md"):
            source_id = file_path.stem
            try:
                content = file_path.read_text(encoding='utf-8')
                # Reuse the regex extractor from MarkdownProcessor
                links = processor.extract_links(content)
                
                for target_id in links:
                    edges.append({
                        "source": source_id,
                        "target": target_id
                    })
                    
                    # Track "ghost notes" (linked but not yet created)
                    if target_id not in existing_files:
                        ghost_nodes.add(target_id)
            except Exception as e:
                logger.warning(f"Could not parse file {file_path.name} for graph: {e}")
                
        # Step 3: Add ghost notes to the nodes list so they appear in the graph
        for ghost in ghost_nodes:
            nodes.append({
                "id": ghost,
                "label": ghost,
                "group": "ghost"
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        }
        
    except Exception as e:
        logger.error(f"Error generating graph data: {e}")
        raise HTTPException(status_code=500, detail="Could not generate graph data")
