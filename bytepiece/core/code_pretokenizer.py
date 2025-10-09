"""Code-aware pre-tokenization for Python source code.

This module provides syntax-aware tokenization specifically for Python,
ensuring that strings, comments, and operators are not broken during
pre-tokenization.

Why this matters:
-----------------
Standard BPE breaks Python syntax:
- "https://api.com" → ["https", "://", "api", ".", "com"]
- x >= 100 → ["x", " >=", " ", "100"]  # >= might be broken

Python-aware pre-tokenization preserves syntax:
- String literals: "https://api.com" → ['"https://api.com"']
- Operators: >= → [">="]
- Comments: # comment → ["# comment"]
"""

import re
from typing import List


class CodePreTokenizer:
    OPERATORS = [
        
        '==', '!=', '<=', '>=',
        
        '+=', '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=', '^=', '>>=', '<<=',
       
        '//', '**', '->', '...',
    ]
    

    STRING_PATTERN = re.compile(
        r'""".*?"""|'  
        r"'''.*?'''|"  
        r'f?"[^"\\]*(?:\\.[^"\\]*)*"|'  
        r"f?'[^'\\]*(?:\\.[^'\\]*)*'",  
        re.DOTALL
    )
    

    COMMENT_PATTERN = re.compile(r'#.*?$', re.MULTILINE)
    

    IDENTIFIER_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    
 
    NUMBER_PATTERN = re.compile(
        r'0[xX][0-9a-fA-F]+|'  # Hex
        r'0[bB][01]+|'  # Binary
        r'\d+\.\d+|'  # Float
        r'\d+'  # Integer
    )
    
    def __init__(self):

        self.operators_sorted = sorted(self.OPERATORS, key=len, reverse=True)
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        pos = 0
        
        
        protected = []
        
        
        for match in self.STRING_PATTERN.finditer(text):
            protected.append((match.start(), match.end(), match.group()))
        
        
        for match in self.COMMENT_PATTERN.finditer(text):
            protected.append((match.start(), match.end(), match.group()))
        
        
        protected.sort(key=lambda x: x[0])
        
        
        for start, end, content in protected:
           
            if pos < start:
                chunks.extend(self._tokenize_code(text[pos:start]))
            
           
            chunks.append(content)
            pos = end
        
        
        if pos < len(text):
            chunks.extend(self._tokenize_code(text[pos:]))
        
        return [c for c in chunks if c]  
    
    def _tokenize_code(self, text: str) -> List[str]:

        chunks = []
        i = 0
        
        while i < len(text):
           
            if text[i].isspace():
                ws_start = i
                while i < len(text) and text[i].isspace():
                    i += 1
                chunks.append(text[ws_start:i])
                continue
            
            
            matched_op = False
            for op in self.operators_sorted:
                if text[i:i+len(op)] == op:
                    chunks.append(op)
                    i += len(op)
                    matched_op = True
                    break
            
            if matched_op:
                continue
            
           
            id_match = self.IDENTIFIER_PATTERN.match(text, i)
            if id_match:
                chunks.append(id_match.group())
                i = id_match.end()
                continue
            
            
            num_match = self.NUMBER_PATTERN.match(text, i)
            if num_match:
                chunks.append(num_match.group())
                i = num_match.end()
                continue
            
           
            chunks.append(text[i])
            i += 1
        
        return chunks


def code_pretokenize(text: str) -> List[str]:
    tokenizer = CodePreTokenizer()
    return tokenizer.tokenize(text)