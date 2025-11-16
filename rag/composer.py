"""
Prompt Composer - Context Assembly and Prompt Generation
========================================================

This module handles:
- Formatting retrieved documents into context
- Assembling prompts from question + context
- Context truncation and optimization
- Multiple prompt templates (academic, concise, detailed)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag.retriever import RetrievalResult
except ImportError:
    # Fallback for type hints
    from typing import Any as RetrievalResult


@dataclass
class PromptConfig:
    """Configuration for prompt composition"""
    max_context_length: int = 4000  # Maximum characters in context
    include_metadata: bool = True  # Include paper metadata (title, authors, etc.)
    context_style: str = "detailed"  # detailed, concise, minimal
    citation_style: str = "numeric"  # numeric ([1]), inline (Author, Year)
    truncation_strategy: str = "smart"  # smart, simple, none


class ContextFormatter:
    """Format retrieved documents into context"""

    @staticmethod
    def format_source(
        result: RetrievalResult,
        index: int,
        style: str = "detailed",
        include_metadata: bool = True
    ) -> str:
        """
        Format a single source document

        Args:
            result: Retrieved document
            index: Source number (for citation)
            style: Formatting style
            include_metadata: Whether to include paper metadata

        Returns:
            Formatted source text
        """
        content = result.content

        if style == "detailed":
            # Full format with metadata
            title = result.metadata.get('title', 'Unknown Paper')
            section = result.metadata.get('section', '')

            if include_metadata:
                header = f"[{index}] {title}"
                if section:
                    header += f" (Section: {section})"
                return f"{header}\n{content}"
            else:
                return f"[{index}] {content}"

        elif style == "concise":
            # Compact format
            title = result.metadata.get('title', 'Unknown')
            return f"[{index}] {content}\n(Source: {title})"

        elif style == "minimal":
            # Minimal format (content only)
            return f"[{index}] {content}"

        else:
            raise ValueError(f"Unknown style: {style}")

    @staticmethod
    def format_context(
        results: List[RetrievalResult],
        style: str = "detailed",
        include_metadata: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format all retrieved documents into context

        Args:
            results: List of retrieved documents
            style: Formatting style
            include_metadata: Whether to include metadata
            max_length: Maximum context length (characters)

        Returns:
            Formatted context string
        """
        if not results:
            return "[No relevant documents found]"

        context_parts = []

        for idx, result in enumerate(results, 1):
            formatted = ContextFormatter.format_source(
                result, idx, style, include_metadata
            )
            context_parts.append(formatted)

            # Check length if truncation enabled
            if max_length:
                current_length = sum(len(p) for p in context_parts)
                if current_length > max_length:
                    # Truncate last part
                    excess = current_length - max_length
                    if excess < len(formatted):
                        context_parts[-1] = formatted[:-excess] + "..."
                    else:
                        context_parts.pop()  # Remove entire last part
                    break

        return "\n\n".join(context_parts)


class PromptTemplate:
    """Prompt templates for different use cases"""

    @staticmethod
    def academic_template(question: str, context: str) -> str:
        """
        Academic research assistant template

        Emphasizes:
        - Citation accuracy
        - Technical precision
        - Evidence-based answers
        - Synthesized understanding (not copying)
        """
        return f"""Answer the following question based on the research paper excerpts provided below.

Context from research papers:
{context}

Question: {question}

Instructions:
- SYNTHESIZE information from the context in your own words - DO NOT copy sentences verbatim
- REPHRASE the key ideas while maintaining technical accuracy
- ALWAYS cite sources using [1], [2], etc. when referencing information
- Answer based ONLY on the information in the context above
- Provide a coherent, well-structured explanation that integrates information from multiple sources
- Include a brief concluding sentence that synthesizes the main point
- If the context doesn't fully answer the question, state the limitations
- Do NOT include prompt text or instructions in your answer

Answer:"""

    @staticmethod
    def concise_template(question: str, context: str) -> str:
        """
        Concise answer template

        Emphasizes:
        - Brevity
        - Direct answers
        - Key points only
        """
        return f"""Answer the question based on the provided research excerpts.

Context:
{context}

Question: {question}

Instructions:
- Provide a concise, direct answer
- Cite sources using [1], [2], etc.
- Focus on key points only
- Maximum 3-4 sentences

Answer:"""

    @staticmethod
    def detailed_template(question: str, context: str) -> str:
        """
        Detailed explanation template

        Emphasizes:
        - Comprehensive coverage
        - Multiple perspectives
        - In-depth analysis
        """
        return f"""Provide a detailed answer based on the research papers provided.

Research Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive, well-structured answer
- Cover multiple aspects and perspectives from the papers
- Cite specific sources [1], [2], etc. for each claim
- Include relevant details, examples, and explanations
- Organize your answer with clear paragraphs or sections
- Acknowledge any gaps or limitations in the available research

Detailed Answer:"""

    @staticmethod
    def comparative_template(question: str, context: str) -> str:
        """
        Comparative analysis template

        For questions comparing concepts, methods, or papers
        """
        return f"""Compare and contrast based on the provided research papers.

Research Papers:
{context}

Question: {question}

Instructions:
- Compare the relevant concepts, methods, or findings across papers
- Highlight key similarities and differences
- Cite specific sources [1], [2], etc. for each point
- Present findings in a structured format (e.g., bullet points or table-like)
- Be objective and evidence-based

Comparison:"""

    @staticmethod
    def definition_template(question: str, context: str) -> str:
        """
        Definition-focused template with structured answer format

        For "What is X?" type questions
        Enforces structured output: Definition → Mechanism → Application → Summary
        """
        return f"""Provide a comprehensive explanation of the concept based on the research papers below.

Research Context:
{context}

Question: {question}

Instructions:
Structure your answer in the following format:

**Definition:**
- SYNTHESIZE a clear, concise definition by integrating information from the sources
- REPHRASE key concepts in your own words - DO NOT copy sentences verbatim
- Cite the sources [1], [2], etc.

**Mechanism/Architecture:**
- EXPLAIN how it works using your own phrasing
- Integrate technical details from multiple papers
- Cite sources for each claim [1], [2], etc.

**Applications:**
- DESCRIBE practical applications by synthesizing information
- Rephrase use cases in your own words
- Cite relevant sources [1], [2], etc.

**Summary:**
- Provide a 1-2 sentence synthesis of the main concept and its significance

Remember:
- Base your answer ONLY on the provided context
- SYNTHESIZE and REPHRASE - copying verbatim is NOT allowed
- ALWAYS cite sources using [1], [2], etc.
- Be technical and precise while using your own words

Answer:"""

    @staticmethod
    def afel_template(question: str, context: str) -> str:
        """
        A.F.E.L. Template - Answer, Facts, Evidence, Links

        Simplified version to prevent prompt leakage and ensure proper formatting.
        Detects language and provides appropriate instructions.
        """
        # Detect if question is in Chinese
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in question)

        if is_chinese:
            # Simplified Chinese template for Ollama
            return f"""你是一个学术助手。请用中文回答下面的问题。

问题：{question}

参考资料：
{context}

要求：
1. 用2-3段话直接回答问题
2. 用自己的话解释，不要复制原文
3. 引用来源时用 [1]、[2] 等标注
4. 回答要清晰、准确、简洁

回答："""
        else:
            # Simplified English template for Ollama
            return f"""You are an academic assistant. Answer the question based on the research papers below.

Question: {question}

Research Papers:
{context}

Requirements:
1. Answer in 2-3 clear paragraphs
2. Explain in your own words, don't copy text verbatim
3. Cite sources using [1], [2], etc.
4. Be clear, accurate, and concise

Answer:"""

    @staticmethod
    def general_knowledge_fallback_template(question: str) -> str:
        """
        Template for general knowledge questions when retrieval fails

        Used when:
        - Query is general ("What is AI?")
        - Retrieved papers are irrelevant (low scores)
        - User needs conceptual knowledge, not specific research
        """
        return f"""You are an AI research assistant. The user has asked a general knowledge question that may not require specific research papers.

Question: {question}

INSTRUCTIONS:
Since this appears to be a general knowledge question, provide a clear, accurate answer based on established knowledge in the field.

Structure your answer as follows:

1. **Direct Answer:**
   - Provide a clear, concise definition or explanation
   - Use standard terminology and concepts from the field
   - Be technically accurate but accessible

2. **Key Concepts:**
   - Explain 2-3 fundamental concepts related to the question
   - Use examples where helpful

3. **Context:**
   - Briefly mention the broader significance or applications
   - Connect to current state of the field if relevant

4. **Note:**
   - Add: "Note: This answer is based on general knowledge in the field. For specific research findings, please ask about particular papers or methods."

IMPORTANT:
- Do NOT make up citations or paper references
- Do NOT pretend to quote from papers
- Be honest that this is based on general knowledge
- Be accurate and helpful

Answer:

"""


class ContextSummarizer:
    """Summarize and fuse context from multiple sources"""

    @staticmethod
    def extract_key_points(results: List[RetrievalResult], max_points: int = 5) -> List[Dict]:
        """
        Extract key points from retrieved documents

        Args:
            results: Retrieved documents
            max_points: Maximum key points to extract

        Returns:
            List of key points with metadata
        """
        key_points = []

        for idx, result in enumerate(results[:max_points], 1):
            # Extract first substantial sentence (>30 chars)
            sentences = [s.strip() for s in result.content.split('.') if len(s.strip()) > 30]

            if sentences:
                key_points.append({
                    'text': sentences[0],
                    'source_index': idx,
                    'title': result.metadata.get('title', 'Unknown'),
                    'score': max(result.rerank_score, result.combined_score, result.vector_score)
                })

        return key_points

    @staticmethod
    def create_fusion_summary(key_points: List[Dict]) -> str:
        """
        Create a fusion summary from key points

        Args:
            key_points: Extracted key points

        Returns:
            Fused summary text
        """
        if not key_points:
            return ""

        summary_parts = [
            "Key insights from the research papers:",
            ""
        ]

        for point in key_points:
            summary_parts.append(
                f"- {point['text']} (Source: {point['title'][:50]}...)"
            )

        summary_parts.append("")
        summary_parts.append(
            "Note: The above represents the main findings. Synthesize these insights in your answer."
        )

        return "\n".join(summary_parts)


class PromptComposer:
    """Main prompt composition orchestrator"""

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt composer

        Args:
            config: Prompt configuration (uses defaults if None)
        """
        self.config = config or PromptConfig()
        self.summarizer = ContextSummarizer()

    def compose(
        self,
        question: str,
        results: List[RetrievalResult],
        template_style: str = "academic",
        custom_template: Optional[str] = None,
        add_fusion_summary: bool = False
    ) -> str:
        """
        Compose complete prompt from question and results

        Args:
            question: User question
            results: Retrieved documents
            template_style: Template to use (academic, concise, detailed, comparative, definition)
            custom_template: Custom template string (overrides template_style)
            add_fusion_summary: Add a fusion summary before context

        Returns:
            Complete formatted prompt ready for LLM
        """
        # Optionally add fusion summary
        context_prefix = ""
        if add_fusion_summary:
            key_points = self.summarizer.extract_key_points(results)
            if key_points:
                fusion_summary = self.summarizer.create_fusion_summary(key_points)
                context_prefix = fusion_summary + "\n\n" + "="*60 + "\n\n"

        # Format context
        context = ContextFormatter.format_context(
            results=results,
            style=self.config.context_style,
            include_metadata=self.config.include_metadata,
            max_length=self.config.max_context_length
        )

        # Combine prefix and context
        full_context = context_prefix + context

        # Select template
        if custom_template:
            prompt = custom_template.format(context=full_context, question=question)
        else:
            prompt = self._get_template(template_style, question, full_context)

        return prompt

    def _get_template(self, style: str, question: str, context: str) -> str:
        """Get prompt template by style"""
        templates = {
            'academic': PromptTemplate.academic_template,
            'concise': PromptTemplate.concise_template,
            'detailed': PromptTemplate.detailed_template,
            'comparative': PromptTemplate.comparative_template,
            'definition': PromptTemplate.definition_template,
            'afel': PromptTemplate.afel_template,  # New A.F.E.L. template
            'general_fallback': lambda q, c: PromptTemplate.general_knowledge_fallback_template(q)
        }

        template_fn = templates.get(style)
        if not template_fn:
            raise ValueError(f"Unknown template style: {style}")

        return template_fn(question, context)

    def compose_empty_context_response(self, question: str) -> str:
        """
        Generate safe response when no relevant documents found

        Args:
            question: User question

        Returns:
            Informative response explaining the situation
        """
        return f"""I apologize, but I couldn't find relevant information in the academic paper database to answer your question: "{question}"

**This could mean:**
- The question is outside the scope of the indexed papers
- The query terms don't match the available content
- The papers relevant to this topic haven't been indexed yet
- The question may be too specific or use different terminology

**Suggestions:**
- Try rephrasing your question with different keywords
- Use more general terms related to your topic
- Ask about topics more directly covered in machine learning/AI research papers
- Check if the question is within the domain of computer science research

**Note:** This response is based on a search of the indexed academic papers. I cannot provide information beyond what's in the database.

If you believe this is an error, please try reformulating your question or contact the system administrator to verify the paper index is up to date."""

    def truncate_context(
        self,
        context: str,
        max_length: int,
        strategy: str = "smart"
    ) -> str:
        """
        Truncate context to fit within length limit

        Args:
            context: Full context string
            max_length: Maximum allowed length
            strategy: Truncation strategy (smart, simple, none)

        Returns:
            Truncated context
        """
        if len(context) <= max_length:
            return context

        if strategy == "none":
            return context

        elif strategy == "simple":
            # Simple truncation with ellipsis
            return context[:max_length-3] + "..."

        elif strategy == "smart":
            # Smart truncation: preserve complete sources
            sources = context.split("\n\n")
            truncated_sources = []
            current_length = 0

            for source in sources:
                if current_length + len(source) <= max_length:
                    truncated_sources.append(source)
                    current_length += len(source) + 2  # +2 for \n\n
                else:
                    # Add partial last source if space allows
                    remaining = max_length - current_length
                    if remaining > 100:  # Only if meaningful content fits
                        truncated_sources.append(source[:remaining-3] + "...")
                    break

            result = "\n\n".join(truncated_sources)

            # Add truncation notice
            total_sources = len(sources)
            kept_sources = len(truncated_sources)
            if kept_sources < total_sources:
                result += f"\n\n[Note: Showing {kept_sources} of {total_sources} sources due to length limits]"

            return result

        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")


# ============================================================================
# Convenience Functions
# ============================================================================

def compose_prompt(
    question: str,
    results: List[RetrievalResult],
    template_style: str = "academic",
    max_context_length: int = 4000
) -> str:
    """
    Convenience function to compose prompt

    Args:
        question: User question
        results: Retrieved documents
        template_style: Template style to use
        max_context_length: Maximum context length

    Returns:
        Formatted prompt
    """
    config = PromptConfig(max_context_length=max_context_length)
    composer = PromptComposer(config)
    return composer.compose(question, results, template_style)


def compose_empty_response(question: str) -> str:
    """Convenience function for empty context response"""
    composer = PromptComposer()
    return composer.compose_empty_context_response(question)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'PromptConfig',
    'ContextFormatter',
    'PromptTemplate',
    'ContextSummarizer',
    'PromptComposer',
    'compose_prompt',
    'compose_empty_response'
]
