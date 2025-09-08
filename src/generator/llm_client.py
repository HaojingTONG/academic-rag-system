# src/generator/llm_client.py
"""
LLM客户端 - 与本地/远程语言模型通信
支持Ollama、OpenAI API等多种模型
"""

import json
import requests
import subprocess
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
            answer = f"很抱歉，我无法在当前的学术文献中找到关于\"{query}\"的具体信息。建议您：\n\n1. 尝试使用不同的关键词重新搜索\n2. 扩展您的查询范围\n3. 查看相关领域的最新研究进展"
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
        """生成定义类回答"""
        concept = self._extract_main_concept(query)
        
        answer_parts = [f"根据相关学术文献，{concept}是一个重要的概念，具有以下特点：\n"]
        
        # 整合主要观点为连贯的定义
        if len(main_points) >= 1:
            answer_parts.append(f"**核心定义**：{main_points[0]}\n")
        
        if len(main_points) >= 2:
            answer_parts.append(f"**主要特征**：{main_points[1]}\n")
        
        if len(main_points) >= 3:
            answer_parts.append(f"**应用场景**：{main_points[2]}\n")
        
        # 添加总结
        answer_parts.append(f"总的来说，{concept}在相关领域中扮演着重要角色，其研究和应用具有重要的学术和实用价值。")
        
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
        """生成通用回答"""
        answer_parts = [f"基于相关学术文献，关于\"{query}\"的研究表明：\n"]
        
        # 生成连贯的回答而不是简单列表
        if len(main_points) >= 1:
            answer_parts.append(f"**主要发现**：{main_points[0]}\n")
        
        if len(main_points) >= 2:
            answer_parts.append(f"**研究进展**：{main_points[1]}\n")
        
        if len(main_points) >= 3:
            answer_parts.append(f"**实践应用**：{main_points[2]}\n")
        
        # 如果有更多信息，添加额外观点
        if len(main_points) >= 4:
            answer_parts.append("**其他重要观点**：")
            for point in main_points[3:6]:  # 最多再加3个
                answer_parts.append(f"• {point}")
            answer_parts.append("")
        
        answer_parts.append("这些研究为该领域的理论发展和实际应用提供了重要参考。")
        
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
    
    def __init__(self, preferred_model: str = "llama3.1:8b"):
        self.ollama_client = OllamaClient(default_model=preferred_model)
        self.fallback_generator = FallbackGenerator()
        self.preferred_model = preferred_model
        
        # 检查可用性
        self.ollama_available = self.ollama_client.is_available()
        
        print(f"LLM管理器初始化:")
        print(f"  Ollama可用: {self.ollama_available}")
        if self.ollama_available:
            print(f"  可用模型: {self.ollama_client.available_models}")
        print(f"  后备生成器: 已加载")
    
    def generate_answer(self, prompt: str, query_intent: str = 'general',
                       max_tokens: int = 512, temperature: float = 0.7) -> LLMResponse:
        """生成回答 - 自动选择最佳可用方法"""
        
        # 优先使用Ollama
        if self.ollama_available:
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
        """从prompt中解析查询和上下文"""
        lines = prompt.split('\n')
        query = ""
        context = ""
        
        for line in lines:
            if line.startswith('Query:') or line.startswith('用户问题:') or line.startswith('问题:'):
                query = line.split(':', 1)[1].strip()
            elif line.startswith('Context:') or line.startswith('上下文:') or line.startswith('学术资源:'):
                context = line.split(':', 1)[1].strip()
            elif 'Source:' in line or '内容:' in line:
                context += " " + line
        
        return query or "未知问题", context or prompt

# 单例管理器
_llm_manager = None

def get_llm_manager(preferred_model: str = "llama3.1:8b") -> LLMManager:
    """获取LLM管理器实例"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager(preferred_model)
    return _llm_manager

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