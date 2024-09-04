import sys
from custom_logging import logger
from custom_exception import CustomException

def format_docs(docs):
    """
    Format documents for retrieval output.

    Args:
        docs (list): List of document objects.

    Returns:
        str: Formatted documents as a single string.
    """
    try:
        formatted_docs = []
        for doc in docs:
            # Format the metadata into a string
            metadata_str = ', '.join(f"{key}: {value}" for key, value in doc.metadata.items())

            # Combine page content with its metadata
            doc_str = f"{doc.page_content}\nMetadata: {metadata_str}"

            # Append to the list of formatted documents
            formatted_docs.append(doc_str)

        # Join all formatted documents with double newlines
        logger.info("Documents formatted successfully")
        return "\n\n".join(formatted_docs)
    except Exception as e:
        raise CustomException(e, sys)
