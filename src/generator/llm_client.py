# src/generator/llm_client.py
"""
LLM客户端 - 与本地/远程语言模型通信
支持Ollama、OpenAI API等多种模型
"""

import json
import requests
import subprocess
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM响应结果"""
    text: str
    success: bool
    error_message: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None

class OllamaClient:
    """Ollama本地LLM客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama2"):
        self.base_url = base_url
        self.default_model = default_model
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # 解析输出获取模型名称
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                print("Ollama未运行或无可用模型")
                return []
        except Exception as e:
            print(f"获取Ollama模型列表失败: {e}")
            return []
    
    def is_available(self) -> bool:
        """检查Ollama是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, model: Optional[str] = None, 
                 max_tokens: int = 512, temperature: float = 0.7) -> LLMResponse:
        """生成文本回答"""
        if not self.is_available():
            return LLMResponse(
                text="Ollama服务未运行，请先启动Ollama",
                success=False,
                error_message="Ollama service not available"
            )
        
        model_name = model or self.default_model
        
        # 如果指定的模型不可用，使用第一个可用模型
        if model_name not in self.available_models and self.available_models:
            model_name = self.available_models[0]
            print(f"模型 {model or self.default_model} 不可用，使用 {model_name}")
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60  # 较长的超时时间用于生成
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    text=result.get("response", "").strip(),
                    success=True,
                    model=model_name
                )
            else:
                return LLMResponse(
                    text=f"API调用失败: {response.status_code}",
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            return LLMResponse(
                text="请求超时，请稍后重试",
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            return LLMResponse(
                text=f"生成失败: {str(e)}",
                success=False,
                error_message=str(e)
            )

class OpenAIClient:
    """OpenAI API客户端"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1",
                 default_model: str = "gpt-4o-mini", organization: Optional[str] = None):
        """
        初始化OpenAI客户端

        Args:
            api_key: OpenAI API密钥 (如果为None，从环境变量OPENAI_API_KEY获取)
            base_url: API基础URL (默认为OpenAI官方API)
            default_model: 默认模型名称
            organization: OpenAI组织ID (可选)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.organization = organization

        if not self.api_key:
            print("⚠️ 警告: OPENAI_API_KEY未设置，请设置环境变量或传入api_key参数")

    def is_available(self) -> bool:
        """检查OpenAI API是否可用"""
        if not self.api_key:
            return False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"OpenAI API检查失败: {e}")
            return False

    def generate(self, prompt: str, model: Optional[str] = None,
                 max_tokens: int = 2000, temperature: float = 0.1,
                 system_message: Optional[str] = None) -> LLMResponse:
        """
        生成文本回答

        Args:
            prompt: 用户提示
            model: 模型名称 (如果为None使用默认模型)
            max_tokens: 最大token数
            temperature: 温度参数
            system_message: 系统消息 (可选)

        Returns:
            LLMResponse对象
        """
        if not self.api_key:
            return LLMResponse(
                text="OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量",
                success=False,
                error_message="API key not configured"
            )

        model_name = model or self.default_model

        try:
            # 构建消息列表
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # 构建请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # 发送请求
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                tokens_used = result.get('usage', {}).get('total_tokens', None)

                return LLMResponse(
                    text=answer,
                    success=True,
                    model=model_name,
                    tokens_used=tokens_used
                )
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")

                return LLMResponse(
                    text=f"OpenAI API调用失败: {error_msg}",
                    success=False,
                    error_message=error_msg
                )

        except requests.exceptions.Timeout:
            return LLMResponse(
                text="OpenAI API请求超时，请稍后重试",
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            return LLMResponse(
                text=f"OpenAI API调用异常: {str(e)}",
                success=False,
                error_message=str(e)
            )

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        if not self.api_key:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                models = [m['id'] for m in data.get('data', [])]
                # 只返回常用的GPT模型
                return [m for m in models if m.startswith(('gpt-', 'o1-'))]
            else:
                return []

        except Exception as e:
            print(f"获取OpenAI模型列表失败: {e}")
            return []

class FallbackGenerator:
    """后备生成器 - 当LLM不可用时使用规则生成"""
    
    def __init__(self):
        self.templates = {
            'definition': "根据检索到的文献，{concept}是指{definition}。主要特点包括：{features}。",
            'comparison': "基于现有文献分析，{item1}和{item2}的主要区别在于：{differences}。",
            'methodology': "根据相关研究，{method}的实现步骤包括：{steps}。",
            'general': "基于检索到的学术资料，{answer}。相关研究表明{evidence}。"
        }
    
    def generate_fallback_answer(self, query: str, context_content: str, 
                                query_intent: str = 'general') -> LLMResponse:
        """生成后备回答 - 模拟ChatGPT风格的回答"""
        
        # 提取关键信息
        key_info = self._extract_key_info(context_content)
        main_points = key_info.get('main_points', [])
        
        if not main_points:
            answer = f'I apologize, but I could not find specific information about "{query}" in the current academic literature. Suggestions:\n\n1. Try rephrasing your question with different keywords\n2. Expand your query scope\n3. Check recent research developments in related areas'
        else:
            # 根据查询意图生成连贯的回答
            if query_intent == 'definition' or 'what is' in query.lower() or '什么是' in query:
                answer = self._generate_definition_answer(query, main_points)
            elif query_intent == 'comparison' or 'difference' in query.lower() or '区别' in query or 'compare' in query.lower():
                answer = self._generate_comparison_answer(query, main_points)
            elif query_intent == 'methodology' or 'how to' in query.lower() or '如何' in query or 'method' in query.lower():
                answer = self._generate_methodology_answer(query, main_points)
            else:
                answer = self._generate_general_answer(query, main_points)
        
        return LLMResponse(
            text=answer,
            success=True,
            model="enhanced_fallback_generator"
        )
    
    def _generate_definition_answer(self, query: str, main_points: List[str]) -> str:
        """Generate definition-type answer with structured format"""
        concept = self._extract_main_concept(query)

        answer_parts = []

        # Definition section
        if len(main_points) >= 1:
            answer_parts.append(f"**Definition:**\n{main_points[0]} [1]\n")
        else:
            answer_parts.append(f"**Definition:**\nBased on the research papers, {concept} is a key concept in this field. [1]\n")

        # Mechanism/Architecture section
        if len(main_points) >= 2:
            answer_parts.append(f"**Mechanism/Architecture:**\n{main_points[1]} [2]\n")
        else:
            answer_parts.append(f"**Mechanism/Architecture:**\nThe mechanism involves several key components as described in the literature. [2]\n")

        # Applications section
        if len(main_points) >= 3:
            answer_parts.append(f"**Applications:**\n{main_points[2]} [3]\n")
        else:
            answer_parts.append(f"**Applications:**\nThis concept has various applications in research and practice. [3]\n")

        # Note about sources
        answer_parts.append("\nNote: This answer is generated from the retrieved research papers. For complete details, please refer to the original sources.")

        return "\n".join(answer_parts)
    
    def _generate_comparison_answer(self, query: str, main_points: List[str]) -> str:
        """生成比较类回答"""
        answer_parts = [f"基于现有学术文献，关于\"{query}\"的比较分析如下：\n"]
        
        for i, point in enumerate(main_points[:3], 1):
            if '与' in point or 'and' in point.lower() or 'vs' in point.lower():
                answer_parts.append(f"**对比点{i}**：{point}\n")
            else:
                answer_parts.append(f"**关键差异{i}**：{point}\n")
        
        answer_parts.append("通过文献分析可以看出，不同方法各有其优势和适用场景，选择时应根据具体需求和条件进行权衡。")
        
        return "\n".join(answer_parts)
    
    def _generate_methodology_answer(self, query: str, main_points: List[str]) -> str:
        """生成方法类回答"""
        answer_parts = [f"根据相关研究文献，关于\"{query}\"的方法和步骤包括：\n"]
        
        for i, point in enumerate(main_points[:4], 1):
            # 尝试识别方法步骤
            if any(word in point.lower() for word in ['first', 'then', 'next', 'finally', '首先', '然后', '接下来', '最后']):
                answer_parts.append(f"**步骤{i}**：{point}\n")
            elif any(word in point.lower() for word in ['method', 'approach', 'technique', '方法', '技术']):
                answer_parts.append(f"**核心方法**：{point}\n")
            else:
                answer_parts.append(f"**要点{i}**：{point}\n")
        
        answer_parts.append("这些方法在实际应用中需要根据具体情况进行调整和优化，以达到最佳效果。")
        
        return "\n".join(answer_parts)
    
    def _generate_general_answer(self, query: str, main_points: List[str]) -> str:
        """Generate general answer with English format"""
        answer_parts = [f'Based on the research papers, regarding "{query}":\n']

        # Generate coherent answer rather than simple list
        if len(main_points) >= 1:
            answer_parts.append(f"{main_points[0]} [1]\n")

        if len(main_points) >= 2:
            answer_parts.append(f"{main_points[1]} [2]\n")

        if len(main_points) >= 3:
            answer_parts.append(f"{main_points[2]} [3]\n")

        # If more information, add additional points
        if len(main_points) >= 4:
            answer_parts.append("\n**Additional Findings:**")
            for i, point in enumerate(main_points[3:6], 4):  # Max 3 more
                answer_parts.append(f"• {point} [{i}]")
            answer_parts.append("")

        answer_parts.append("\nThese research findings provide important reference for both theoretical development and practical applications in the field.")

        return "\n".join(answer_parts)
    
    def _extract_key_info(self, content: str) -> Dict:
        """提取关键信息"""
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        return {
            'main_points': sentences[:5],
            'features': [s for s in sentences if any(word in s.lower() 
                        for word in ['method', 'approach', 'technique', 'feature'])],
        }
    
    def _extract_main_concept(self, query: str) -> str:
        """提取主要概念"""
        # 简单的关键词提取
        words = query.lower().split()
        concepts = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'why']]
        return concepts[0] if concepts else "该概念"
    
    def _extract_comparison_items(self, query: str) -> List[str]:
        """提取比较项"""
        # 简单的比较项提取
        words = query.split()
        return [w for w in words if len(w) > 3][:2]

class LLMManager:
    """LLM管理器 - 统一管理多种LLM客户端"""

    def __init__(self, backend: str = "openai", preferred_model: Optional[str] = None,
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 organization: Optional[str] = None, ollama_host: Optional[str] = None):
        """
        初始化LLM管理器

        Args:
            backend: LLM后端 ("openai" 或 "ollama")
            preferred_model: 首选模型名称
            api_key: OpenAI API密钥 (仅OpenAI后端)
            api_base: OpenAI API基础URL (仅OpenAI后端)
            organization: OpenAI组织ID (仅OpenAI后端)
            ollama_host: Ollama服务地址 (仅Ollama后端)
        """
        self.backend = backend.lower()
        self.fallback_generator = FallbackGenerator()

        # 初始化客户端
        self.openai_client = None
        self.ollama_client = None
        self.openai_available = False
        self.ollama_available = False

        if self.backend == "openai":
            # 初始化OpenAI客户端
            default_model = preferred_model or "gpt-4o-mini"
            self.openai_client = OpenAIClient(
                api_key=api_key,
                base_url=api_base or "https://api.openai.com/v1",
                default_model=default_model,
                organization=organization
            )
            self.preferred_model = default_model
            self.openai_available = self.openai_client.is_available()

            print(f"LLM管理器初始化:")
            print(f"  后端: OpenAI")
            print(f"  OpenAI可用: {self.openai_available}")
            if self.openai_available:
                print(f"  默认模型: {default_model}")
            print(f"  后备生成器: 已加载")

        elif self.backend == "ollama":
            # 初始化Ollama客户端
            default_model = preferred_model or "llama3.1:8b"
            ollama_url = ollama_host or "http://localhost:11434"
            self.ollama_client = OllamaClient(
                base_url=ollama_url,
                default_model=default_model
            )
            self.preferred_model = default_model
            self.ollama_available = self.ollama_client.is_available()

            print(f"LLM管理器初始化:")
            print(f"  后端: Ollama")
            print(f"  Ollama可用: {self.ollama_available}")
            if self.ollama_available:
                print(f"  可用模型: {self.ollama_client.available_models}")
            print(f"  后备生成器: 已加载")

        else:
            raise ValueError(f"不支持的后端: {backend}，请使用 'openai' 或 'ollama'")
    
    def generate_answer(self, prompt: str, query_intent: str = 'general',
                       max_tokens: int = 2000, temperature: float = 0.1) -> LLMResponse:
        """生成回答 - 自动选择最佳可用方法"""

        # 根据后端选择客户端
        if self.backend == "openai" and self.openai_available:
            response = self.openai_client.generate(
                prompt,
                model=self.preferred_model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if response.success and response.text.strip():
                return response
            else:
                print(f"OpenAI生成失败，回退到后备生成器: {response.error_message}")

        elif self.backend == "ollama" and self.ollama_available:
            response = self.ollama_client.generate(
                prompt,
                model=self.preferred_model,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if response.success and response.text.strip():
                return response
            else:
                print(f"Ollama生成失败，回退到后备生成器: {response.error_message}")

        # 后备方案：使用规则生成器
        # 从prompt中提取查询和上下文
        query, context = self._parse_prompt(prompt)
        return self.fallback_generator.generate_fallback_answer(query, context, query_intent)
    
    def _parse_prompt(self, prompt: str) -> tuple:
        """Parse query and context from prompt"""
        lines = prompt.split('\n')
        query = ""
        context = ""
        in_context = False

        for i, line in enumerate(lines):
            # Match various question patterns
            if any(pattern in line for pattern in ['Question:', 'Query:', 'question:', 'query:', '用户问题:', '问题:']):
                query = line.split(':', 1)[1].strip() if ':' in line else ""

            # Match context patterns
            elif any(pattern in line for pattern in ['Context from research', 'Research Context:', 'Context:', '上下文:', '学术资源:']):
                in_context = True
                # Get content after the colon if present
                if ':' in line:
                    context_start = line.split(':', 1)[1].strip()
                    if context_start:
                        context = context_start
                continue

            # If we're in context section, collect all lines
            elif in_context:
                # Stop at Instructions section
                if 'Instructions:' in line or 'instructions:' in line:
                    in_context = False
                    break
                # Accumulate context
                if line.strip():
                    context += " " + line.strip()

        # Clean up
        query = query.strip() if query else "Unknown question"
        context = context.strip() if context else prompt

        return query, context

# 单例管理器
_llm_manager = None

def get_llm_manager(backend: str = "openai", preferred_model: Optional[str] = None,
                   api_key: Optional[str] = None, api_base: Optional[str] = None,
                   organization: Optional[str] = None, ollama_host: Optional[str] = None) -> LLMManager:
    """
    获取LLM管理器实例（单例模式）

    Args:
        backend: LLM后端 ("openai" 或 "ollama")
        preferred_model: 首选模型名称
        api_key: OpenAI API密钥
        api_base: OpenAI API基础URL
        organization: OpenAI组织ID
        ollama_host: Ollama服务地址

    Returns:
        LLMManager实例
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager(
            backend=backend,
            preferred_model=preferred_model,
            api_key=api_key,
            api_base=api_base,
            organization=organization,
            ollama_host=ollama_host
        )
    return _llm_manager

def create_llm_manager_from_config(config=None) -> LLMManager:
    """
    从配置创建LLM管理器

    Args:
        config: 配置对象（如果为None，尝试从configs导入）

    Returns:
        LLMManager实例
    """
    if config is None:
        try:
            from configs import config as global_config
            config = global_config
        except ImportError:
            print("⚠️ 无法导入配置，使用默认设置")
            return LLMManager(backend="openai", preferred_model="gpt-4o-mini")

    generation_config = config.generation

    return LLMManager(
        backend=generation_config.llm_backend,
        preferred_model=generation_config.model,
        api_key=generation_config.openai_api_key,
        api_base=generation_config.openai_api_base,
        organization=generation_config.openai_organization,
        ollama_host=generation_config.ollama_host
    )

# 使用示例和测试
def test_llm_client():
    """测试LLM客户端"""
    print("测试LLM客户端...")
    
    manager = get_llm_manager()
    
    test_prompt = """基于以下学术资料回答问题：

问题：什么是transformer架构？

上下文资料：
Transformer architecture uses self-attention mechanism to process sequences in parallel. 
It consists of encoder and decoder layers with multi-head attention.

请直接回答问题："""
    
    response = manager.generate_answer(test_prompt, query_intent='definition')
    
    print(f"生成成功: {response.success}")
    print(f"使用模型: {response.model}")
    print(f"回答内容: {response.text}")
    if response.error_message:
        print(f"错误信息: {response.error_message}")

if __name__ == "__main__":
    test_llm_client()