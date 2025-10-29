#!/usr/bin/env python3
"""Test OpenAI integration"""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Config module will load .env automatically
from configs import config

print(f'='*60)
print('Testing OpenAI Integration')
print(f'='*60)

print(f'\n1. Environment Variables:')
print(f'   OPENAI_API_KEY set: {bool(os.getenv("OPENAI_API_KEY"))}')
if os.getenv("OPENAI_API_KEY"):
    print(f'   API key prefix: {os.getenv("OPENAI_API_KEY")[:15]}...')

print(f'\n2. Config Values:')
print(f'   LLM Backend: {config.generation.llm_backend}')
print(f'   Model: {config.generation.model}')
print(f'   API Key in config: {bool(config.generation.openai_api_key)}')

print(f'\n3. Creating LLMManager:')
from src.generator.llm_client import create_llm_manager_from_config

manager = create_llm_manager_from_config()
print(f'   Backend: {manager.backend}')
print(f'   OpenAI available: {manager.openai_available}')
if manager.openai_client:
    print(f'   Client API key set: {bool(manager.openai_client.api_key)}')

print(f'\n4. Testing Generation:')
if manager.openai_available:
    print('   Calling OpenAI API...')
    response = manager.generate_answer(
        "Explain what a transformer is in one short sentence.",
        max_tokens=50
    )
    print(f'   Success: {response.success}')
    print(f'   Model: {response.model}')
    print(f'   Response: {response.text}')
else:
    print('   ⚠️ OpenAI not available - would use fallback')
    response = manager.generate_answer("test query")
    print(f'   Fallback response: {response.text[:100]}...')

print(f'\n{'='*60}')
print('Test Complete')
print(f'{'='*60}\n')
