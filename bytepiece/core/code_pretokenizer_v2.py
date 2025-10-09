"""Improved code-aware pre-tokenization (Hybrid approach).

This version balances semantic preservation with compression efficiency:
- Preserves: short strings, all operators, keywords
- Fragments: long strings (intelligently), long numbers, comment content
"""

import re
from typing import List


class ImprovedCodePreTokenizer:
    """Hybrid pre-tokenizer for Python code."""
    
    # High-priority operators (never fragment)
    CRITICAL_OPERATORS = [
        '==', '!=', '<=', '>=', '//', '**', '->', '...',
    ]
    
    # Lower priority (can fragment if needed)
    NORMAL_OPERATORS = [
        '+=', '-=', '*=', '/=', '//=', '%=', '**=', 
        '&=', '|=', '^=', '>>=', '<<=',
    ]
    
    # Threshold for string handling
    SHORT_STRING_THRESHOLD = 30  # chars
    
    # String patterns
    STRING_PATTERN = re.compile(
        r'""".*?"""|'
        r"'''.*?'''|"
        r'f?"[^"\\]*(?:\\.[^"\\]*)*"|'
        r"f?'[^'\\]*(?:\\.[^'\\]*)*'",
        re.DOTALL
    )
    
    # Comment pattern
    COMMENT_PATTERN = re.compile(r'#.*?$', re.MULTILINE)
    
    # Identifier/keyword pattern
    IDENTIFIER_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    
    # Number pattern
    NUMBER_PATTERN = re.compile(
        r'0[xX][0-9a-fA-F]+|'
        r'0[bB][01]+|'
        r'\d+\.\d+|'
        r'\d+'
    )
    
    def __init__(self):
        """Initialize improved tokenizer."""
        self.all_operators = sorted(
            self.CRITICAL_OPERATORS + self.NORMAL_OPERATORS,
            key=len,
            reverse=True
        )
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize with hybrid strategy.
        
        Args:
            text: Python source code
            
        Returns:
            List of tokens balancing semantics and compression
        """
        if not text:
            return []
        
        chunks = []
        pos = 0
        
        # Find strings and comments
        protected = []
        
        # Strings
        for match in self.STRING_PATTERN.finditer(text):
            protected.append((match.start(), match.end(), match.group(), 'string'))
        
        # Comments
        for match in self.COMMENT_PATTERN.finditer(text):
            protected.append((match.start(), match.end(), match.group(), 'comment'))
        
        protected.sort(key=lambda x: x[0])
        
        # Process
        for start, end, content, kind in protected:
            # Code before
            if pos < start:
                chunks.extend(self._tokenize_code(text[pos:start]))
            
            # Handle protected content
            if kind == 'string':
                chunks.extend(self._tokenize_string(content))
            elif kind == 'comment':
                chunks.extend(self._tokenize_comment(content))
            
            pos = end
        
        # Remaining code
        if pos < len(text):
            chunks.extend(self._tokenize_code(text[pos:]))
        
        return [c for c in chunks if c]
    
    def _tokenize_string(self, string: str) -> List[str]:
        """Smart string tokenization.
        
        Short strings: preserve whole
        Long strings: fragment but keep quotes
        """
        # Extract quote type
        if string.startswith('"""') or string.startswith("'''"):
            quotes = string[:3]
            content = string[3:-3]
            is_triple = True
        elif string.startswith('f"') or string.startswith("f'"):
            quotes = string[:2]
            content = string[2:-1]
            is_triple = False
        else:
            quotes = string[0]
            content = string[1:-1]
            is_triple = False
        
        # Short string: keep whole
        if len(content) <= self.SHORT_STRING_THRESHOLD:
            return [string]
        
        # Long string: fragment intelligently
        # Keep quotes separate, fragment content
        tokens = [quotes]
        
        # Fragment content by common patterns
        # URLs, paths, etc.
        content_tokens = self._fragment_string_content(content)
        tokens.extend(content_tokens)
        
        # Closing quote
        tokens.append(quotes[-1] if not is_triple else quotes)
        
        return tokens
    
    def _fragment_string_content(self, content: str) -> List[str]:
        """Fragment string content intelligently.
        
        Preserves common patterns:
        - Protocol patterns (https://, http://)
        - Common words
        - Separators
        """
        # Simple approach: split on common boundaries
        # while preserving common patterns
        
        # Common patterns to preserve
        patterns = [
            r'https?://',  # Protocol
            r'www\.',      # WWW
            r'\w+',        # Words
            r'[^\w\s]+',   # Punctuation
        ]
        
        tokens = []
        pos = 0
        
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        for match in re.finditer(combined_pattern, content):
            if match.start() > pos:
                # Add unmatched content
                tokens.append(content[pos:match.start()])
            tokens.append(match.group())
            pos = match.end()
        
        if pos < len(content):
            tokens.append(content[pos:])
        
        return [t for t in tokens if t]
    
    def _tokenize_comment(self, comment: str) -> List[str]:
        """Tokenize comments.
        
        Keep marker (#), fragment content normally.
        """
        # Keep # separate, fragment rest
        if comment.startswith('#'):
            content = comment[1:].strip()
            if len(content) <= 20:  # Short comment
                return [comment]
            else:
                # Fragment long comment
                return ['#', ' '] + self._tokenize_code(content)
        return [comment]
    
    def _tokenize_code(self, text: str) -> List[str]:
        """Tokenize regular code (not strings/comments)."""
        chunks = []
        i = 0
        
        while i < len(text):
            # Whitespace
            if text[i].isspace():
                ws_start = i
                while i < len(text) and text[i].isspace():
                    i += 1
                chunks.append(text[ws_start:i])
                continue
            
            # Operators
            matched_op = False
            for op in self.all_operators:
                if text[i:i+len(op)] == op:
                    chunks.append(op)
                    i += len(op)
                    matched_op = True
                    break
            
            if matched_op:
                continue
            
            # Identifiers
            id_match = self.IDENTIFIER_PATTERN.match(text, i)
            if id_match:
                chunks.append(id_match.group())
                i = id_match.end()
                continue
            
            # Numbers
            num_match = self.NUMBER_PATTERN.match(text, i)
            if num_match:
                num = num_match.group()
                # Fragment long numbers
                if len(num) > 6:
                    # Split into smaller chunks
                    chunks.extend([num[j:j+3] for j in range(0, len(num), 3)])
                else:
                    chunks.append(num)
                i = num_match.end()
                continue
            
            # Single char
            chunks.append(text[i])
            i += 1
        
        return chunks


def improved_code_pretokenize(text: str) -> List[str]:
    """Convenience function for improved tokenization."""
    tokenizer = ImprovedCodePreTokenizer()
    return tokenizer.tokenize(text)