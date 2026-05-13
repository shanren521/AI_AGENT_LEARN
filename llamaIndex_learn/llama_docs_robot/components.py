from typing_extensions import Callable, Optional

from llama_index.core.utils import globals_helper
from llama_index.core.schema import MetadataMode

class LimitRetrievedNodesLength:
    def __init__(self, limit: int = 3000, tokenizer: Optional[Callable] = None):
        self._tokenizer = tokenizer or globals_helper.punkt_tokenizer
        self.limit = limit

    def postprocess_nodes(self, nodes, query_bundle):
        include_nodes = []
        current_length = 0

        for node in nodes:
            current_length += len(self._tokenizer(node.node.get_content(metadata_mode=MetadataMode.LLM)))
            if current_length > self.limit:
                break

            include_nodes.append(node)
        return include_nodes










