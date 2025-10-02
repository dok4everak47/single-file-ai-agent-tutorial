# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic",
#     "pydantic",
# ]
# ///

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from anthropic import Anthropic  # type: ignore
from pydantic import BaseModel  # type: ignore

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)

# 抑制详细的HTTP日志
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class Tool(BaseModel):
    """工具类，定义AI代理可以使用的工具"""
    name: str  # 工具名称
    description: str  # 工具描述
    input_schema: Dict[str, Any]  # 工具输入模式


class AIAgent:
    """AI代理类，负责与Claude API交互和执行工具操作"""
    
    def __init__(self, api_key: str):
        """初始化AI代理
        
        Args:
            api_key: Anthropic API密钥
        """
        self.client = Anthropic(api_key=api_key)  # 创建Anthropic客户端
        self.messages: List[Dict[str, Any]] = []  # 存储对话历史
        self.tools: List[Tool] = []  # 可用工具列表
        self._setup_tools()  # 设置工具

    def _setup_tools(self):
        """设置可用的工具列表"""
        self.tools = [
            Tool(
                name="read_file",
                description="读取指定路径文件的内容",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要读取的文件路径",
                        }
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_files",
                description="列出指定路径中的所有文件和目录",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要列出的目录路径（默认为当前目录）",
                        }
                    },
                    "required": [],
                },
            ),
            Tool(
                name="edit_file",
                description="通过将old_text替换为new_text来编辑文件。如果文件不存在则创建新文件。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要编辑的文件路径",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "要搜索并替换的文本（留空以创建新文件）",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "用于替换old_text的文本",
                        },
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """执行指定的工具
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
            
        Returns:
            工具执行结果
        """
        logging.info(f"执行工具: {tool_name} 输入: {tool_input}")
        try:
            if tool_name == "read_file":
                return self._read_file(tool_input["path"])
            elif tool_name == "list_files":
                return self._list_files(tool_input.get("path", "."))
            elif tool_name == "edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )
            else:
                return f"未知工具: {tool_name}"
        except Exception as e:
            logging.error(f"执行 {tool_name} 时出错: {str(e)}")
            return f"执行 {tool_name} 时出错: {str(e)}"

    def _read_file(self, path: str) -> str:
        """读取文件内容
        
        Args:
            path: 文件路径
            
        Returns:
            文件内容或错误信息
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"文件 {path} 的内容:\n{content}"
        except FileNotFoundError:
            return f"文件未找到: {path}"
        except Exception as e:
            return f"读取文件时出错: {str(e)}"

    def _list_files(self, path: str) -> str:
        """列出目录中的文件和文件夹
        
        Args:
            path: 目录路径
            
        Returns:
            目录内容列表或错误信息
        """
        try:
            if not os.path.exists(path):
                return f"路径未找到: {path}"

            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[目录]  {item}/")
                else:
                    items.append(f"[文件] {item}")

            if not items:
                return f"空目录: {path}"

            return f"{path} 的内容:\n" + "\n".join(items)
        except Exception as e:
            return f"列出文件时出错: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """编辑文件内容
        
        Args:
            path: 文件路径
            old_text: 要替换的旧文本
            new_text: 替换的新文本
            
        Returns:
            操作结果信息
        """
        try:
            if os.path.exists(path) and old_text:
                # 文件存在且有旧文本，执行替换操作
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_text not in content:
                    return f"文件中未找到文本: {old_text}"

                content = content.replace(old_text, new_text)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"成功编辑 {path}"
            else:
                # 文件不存在或没有旧文本，创建新文件
                # 如果路径包含子目录，则创建目录
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                return f"成功创建 {path}"
        except Exception as e:
            return f"编辑文件时出错: {str(e)}"

    def chat(self, user_input: str) -> str:
        """与AI代理进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI代理的回复
        """
        logging.info(f"用户输入: {user_input}")
        self.messages.append({"role": "user", "content": user_input})

        # 准备工具模式供API使用
        tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self.tools
        ]

        while True:
            try:
                # 调用Claude API
                response = self.client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system="你是一个在终端环境中运行的乐于助人的编程助手。只输出纯文本，不要使用markdown格式，因为你的回复会直接显示在终端中。要简洁但全面，以友好的语气提供清晰实用的建议。不要在回复中使用任何星号字符。",
                    messages=self.messages,
                    tools=tool_schemas,
                )

                # 处理API响应
                assistant_message = {"role": "assistant", "content": []}

                for content in response.content:
                    if content.type == "text":
                        assistant_message["content"].append(
                            {"type": "text", "text": content.text}
                        )
                    elif content.type == "tool_use":
                        assistant_message["content"].append(
                            {
                                "type": "tool_use",
                                "id": content.id,
                                "name": content.name,
                                "input": content.input,
                            }
                        )

                self.messages.append(assistant_message)

                # 执行工具调用
                tool_results = []
                for content in response.content:
                    if content.type == "tool_use":
                        result = self._execute_tool(content.name, content.input)
                        logging.info(
                            f"工具结果: {result[:500]}..."
                        )  # 记录前500个字符
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result,
                            }
                        )

                # 如果有工具结果，继续对话
                if tool_results:
                    self.messages.append({"role": "user", "content": tool_results})
                else:
                    # 没有工具调用，返回最终回复
                    return response.content[0].text if response.content else ""

            except Exception as e:
                return f"错误: {str(e)}"


def main():
    """主函数，程序入口点"""
    parser = argparse.ArgumentParser(
        description="AI代码助手 - 具有文件编辑功能的对话式AI代理"
    )
    parser.add_argument(
        "--api-key", help="Anthropic API密钥（或设置ANTHROPIC_API_KEY环境变量）"
    )
    args = parser.parse_args()

    # 获取API密钥（命令行参数或环境变量）
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "错误: 请通过--api-key参数或ANTHROPIC_API_KEY环境变量提供API密钥"
        )
        sys.exit(1)

    # 创建AI代理实例
    agent = AIAgent(api_key)

    print("AI代码助手")
    print("================")
    print("一个可以读取、列出和编辑文件的对话式AI代理。")
    print("输入'exit'或'quit'结束对话。")
    print()

    # 主对话循环
    while True:
        try:
            user_input = input("你: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if not user_input:
                continue

            print("\n助手: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")
            print()


if __name__ == "__main__":
    main()