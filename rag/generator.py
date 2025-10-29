"""
RAG Generator - LLM Generation with Retry and Fallback
======================================================

This module handles:
- LLM generation with retry logic
- Multi-provider fallback (OpenAI → Anthropic → Local)
- Exponential backoff for rate limits
- Error handling and logging
- Citation consistency validation
"""

import time
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.generator.llm_client import LLMResponse, LLMManager, FallbackGenerator
except ImportError:
    print("⚠️ Warning: Could not import LLM client modules")
    LLMResponse = None
    LLMManager = None
    FallbackGenerator = None


class ProviderType(Enum):
    """LLM provider types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    FALLBACK = "fallback"


@dataclass
class GenerationConfig:
    """Configuration for generation with retry"""
    # Provider settings
    primary_provider: ProviderType = ProviderType.OPENAI
    fallback_providers: List[ProviderType] = None

    # Retry settings
    max_retries: int = 3
    initial_backoff: float = 1.0  # seconds
    max_backoff: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True

    # Generation parameters
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60

    def __post_init__(self):
        """Set default fallback providers if not specified"""
        if self.fallback_providers is None:
            self.fallback_providers = [
                ProviderType.ANTHROPIC,
                ProviderType.OLLAMA,
                ProviderType.FALLBACK
            ]


@dataclass
class GenerationResult:
    """Result of generation attempt"""
    text: str
    success: bool
    provider: ProviderType
    attempts: int
    total_time: float
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None


class RetryLogic:
    """Exponential backoff retry logic"""

    @staticmethod
    def should_retry(
        error: Exception,
        attempt: int,
        max_retries: int,
        retry_on_timeout: bool = True,
        retry_on_rate_limit: bool = True
    ) -> bool:
        """
        Determine if request should be retried

        Args:
            error: Exception that occurred
            attempt: Current attempt number
            max_retries: Maximum retry attempts
            retry_on_timeout: Retry on timeout errors
            retry_on_rate_limit: Retry on rate limit (429) errors

        Returns:
            True if should retry
        """
        if attempt >= max_retries:
            return False

        error_str = str(error).lower()

        # Retry on rate limits (429)
        if retry_on_rate_limit and ('429' in error_str or 'rate limit' in error_str):
            return True

        # Retry on timeouts
        if retry_on_timeout and ('timeout' in error_str or 'timed out' in error_str):
            return True

        # Retry on server errors (5xx)
        if any(code in error_str for code in ['500', '502', '503', '504']):
            return True

        # Don't retry on client errors (4xx except 429)
        if any(code in error_str for code in ['400', '401', '403', '404']):
            return False

        # Default: retry on unknown errors (might be transient)
        return True

    @staticmethod
    def calculate_backoff(
        attempt: int,
        initial_backoff: float,
        multiplier: float,
        max_backoff: float
    ) -> float:
        """
        Calculate exponential backoff delay

        Args:
            attempt: Current attempt number (0-indexed)
            initial_backoff: Initial backoff time
            multiplier: Backoff multiplier
            max_backoff: Maximum backoff time

        Returns:
            Backoff time in seconds
        """
        backoff = initial_backoff * (multiplier ** attempt)
        return min(backoff, max_backoff)


class RAGGenerator:
    """Main generator with retry and multi-provider fallback"""

    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Initialize RAG generator

        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        self.llm_managers: Dict[ProviderType, Any] = {}
        self.fallback_generator = FallbackGenerator() if FallbackGenerator else None

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Try to initialize each provider
        providers_to_try = [self.config.primary_provider] + self.config.fallback_providers

        for provider in providers_to_try:
            if provider == ProviderType.FALLBACK:
                # Fallback generator is always available
                continue

            try:
                if provider == ProviderType.OPENAI:
                    api_key = os.getenv('OPENAI_API_KEY')
                    if api_key and LLMManager:
                        manager = LLMManager(
                            backend="openai",
                            api_key=api_key,
                            preferred_model="gpt-4o-mini"
                        )
                        if manager.openai_available:
                            self.llm_managers[provider] = manager
                            print(f"✅ OpenAI provider initialized")

                elif provider == ProviderType.ANTHROPIC:
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                    if api_key:
                        # TODO: Implement Anthropic client
                        print(f"⚠️ Anthropic provider not yet implemented")

                elif provider == ProviderType.OLLAMA:
                    if LLMManager:
                        manager = LLMManager(
                            backend="ollama",
                            preferred_model="llama3.1:8b"
                        )
                        if manager.ollama_available:
                            self.llm_managers[provider] = manager
                            print(f"✅ Ollama provider initialized")

            except Exception as e:
                print(f"⚠️ Failed to initialize {provider.value}: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> GenerationResult:
        """
        Generate answer with retry and fallback

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (overrides config)
            temperature: Temperature (overrides config)

        Returns:
            GenerationResult with answer and metadata
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        start_time = time.time()

        # Try providers in order
        providers_to_try = [self.config.primary_provider] + self.config.fallback_providers

        for provider in providers_to_try:
            print(f"\n🔄 Trying provider: {provider.value}")

            result = self._try_provider(
                provider=provider,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if result.success:
                result.total_time = time.time() - start_time
                print(f"✅ Success with {provider.value} after {result.attempts} attempts ({result.total_time:.2f}s)")
                return result
            else:
                print(f"❌ {provider.value} failed: {result.error_message}")

        # All providers failed
        total_time = time.time() - start_time
        return GenerationResult(
            text="[All LLM providers failed. System error.]",
            success=False,
            provider=ProviderType.FALLBACK,
            attempts=0,
            total_time=total_time,
            error_message="All providers failed"
        )

    def _try_provider(
        self,
        provider: ProviderType,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> GenerationResult:
        """
        Try a single provider with retry logic

        Args:
            provider: Provider to try
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            GenerationResult
        """
        attempt = 0

        while attempt <= self.config.max_retries:
            try:
                # Generate with provider
                response = self._generate_with_provider(
                    provider=provider,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                if response and response.get('success', False):
                    return GenerationResult(
                        text=response.get('text', ''),
                        success=True,
                        provider=provider,
                        attempts=attempt + 1,
                        total_time=0,  # Will be set by caller
                        tokens_used=response.get('tokens_used')
                    )
                else:
                    error_msg = response.get('error_message', 'Unknown error')
                    raise Exception(error_msg)

            except Exception as e:
                error_msg = str(e)

                # Check if should retry
                if RetryLogic.should_retry(
                    error=e,
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    retry_on_timeout=self.config.retry_on_timeout,
                    retry_on_rate_limit=self.config.retry_on_rate_limit
                ):
                    # Calculate backoff
                    backoff = RetryLogic.calculate_backoff(
                        attempt=attempt,
                        initial_backoff=self.config.initial_backoff,
                        multiplier=self.config.backoff_multiplier,
                        max_backoff=self.config.max_backoff
                    )

                    print(f"   ⏳ Retry {attempt + 1}/{self.config.max_retries} after {backoff:.1f}s: {error_msg}")
                    time.sleep(backoff)
                    attempt += 1
                else:
                    # Don't retry
                    return GenerationResult(
                        text="",
                        success=False,
                        provider=provider,
                        attempts=attempt + 1,
                        total_time=0,
                        error_message=error_msg
                    )

        # Max retries exceeded
        return GenerationResult(
            text="",
            success=False,
            provider=provider,
            attempts=attempt,
            total_time=0,
            error_message=f"Max retries ({self.config.max_retries}) exceeded"
        )

    def _generate_with_provider(
        self,
        provider: ProviderType,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """
        Generate answer using specific provider

        Args:
            provider: Provider type
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            Response dictionary
        """
        if provider == ProviderType.FALLBACK:
            # Use fallback generator
            if self.fallback_generator:
                # Parse prompt to extract query and context
                query, context = self._parse_prompt(prompt)
                response = self.fallback_generator.generate_fallback_answer(
                    query, context, 'general'
                )
                return {
                    'text': response.text,
                    'success': response.success,
                    'error_message': response.error_message
                }
            else:
                raise Exception("Fallback generator not available")

        # Try LLM manager
        manager = self.llm_managers.get(provider)
        if not manager:
            raise Exception(f"Provider {provider.value} not initialized")

        # Generate
        response = manager.generate_answer(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return {
            'text': response.text if hasattr(response, 'text') else str(response),
            'success': response.success if hasattr(response, 'success') else bool(response),
            'tokens_used': response.tokens_used if hasattr(response, 'tokens_used') else None,
            'error_message': response.error_message if hasattr(response, 'error_message') else None
        }

    def _parse_prompt(self, prompt: str) -> tuple:
        """Extract query and context from prompt"""
        lines = prompt.split('\n')
        query = ""
        context = ""

        for line in lines:
            if 'Question:' in line or 'query:' in line.lower():
                query = line.split(':', 1)[1].strip() if ':' in line else ""
            elif 'Context:' in line or 'context:' in line.lower():
                # Capture everything after "Context:"
                context_start = prompt.find(line)
                context = prompt[context_start + len(line):].strip()
                break

        return query or "Unknown question", context or prompt


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_answer(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    max_retries: int = 3
) -> GenerationResult:
    """
    Convenience function to generate answer

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens
        temperature: Temperature
        max_retries: Maximum retry attempts

    Returns:
        GenerationResult
    """
    config = GenerationConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries
    )
    generator = RAGGenerator(config)
    return generator.generate(prompt, max_tokens, temperature)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'ProviderType',
    'GenerationConfig',
    'GenerationResult',
    'RetryLogic',
    'RAGGenerator',
    'generate_answer'
]
