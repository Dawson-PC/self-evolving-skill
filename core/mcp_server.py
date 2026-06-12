"""
mcp_server.py - MCP服务器

基于Model Context Protocol（MCP）的服务器实现，
对外提供skill工具的API接口。
支持的MCP工具：
- skill_create: 创建Skill
- skill_execute: 执行并学习
- skill_analyze: 分析嵌入
- skill_list: 列出Skills
- skill_stats: 系统统计
- skill_save: 持久化保存
- skill_load: 加载
"""

from __future__ import annotations

import json
import sys
import logging
from typing import Any, Dict, List, Optional

from .skill_engine import SelfEvolvingSkillEngine
from .storage import Storage

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP协议服务器

    通过stdin/stdout与客户端（如Claude Desktop）通信。
    使用JSON-RPC风格的消息格式。
    """

    def __init__(
        self,
        engine: Optional[SelfEvolvingSkillEngine] = None,
        storage: Optional[Storage] = None,
    ):
        self.engine = engine or SelfEvolvingSkillEngine()
        self.storage = storage or Storage()
        self.storage.bind_engine(self.engine)
        self._running = False

        # 注册工具处理器
        self._tool_handlers = {
            "skill_create": self._handle_skill_create,
            "skill_execute": self._handle_skill_execute,
            "skill_analyze": self._handle_skill_analyze,
            "skill_list": self._handle_skill_list,
            "skill_stats": self._handle_skill_stats,
            "skill_save": self._handle_skill_save,
            "skill_load": self._handle_skill_load,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取MCP工具定义列表"""
        return [
            {
                "name": "skill_create",
                "description": "创建新的Skill",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill名称"},
                        "description": {"type": "string", "description": "描述"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "skill_execute",
                "description": "执行Skill并触发学习",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string", "description": "Skill ID"},
                        "context": {"type": "object", "description": "执行上下文"},
                        "success": {"type": "boolean", "description": "是否成功"},
                        "value": {"type": "number", "description": "价值评分"},
                    },
                    "required": ["skill_id", "context"],
                },
            },
            {
                "name": "skill_analyze",
                "description": "分析嵌入向量",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "嵌入向量",
                        },
                    },
                    "required": ["embedding"],
                },
            },
            {
                "name": "skill_list",
                "description": "列出所有Skills",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "skill_stats",
                "description": "系统统计信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "skill_save",
                "description": "持久化保存当前状态",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "保存标签"},
                    },
                },
            },
            {
                "name": "skill_load",
                "description": "加载已保存的状态",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "文件路径（可选，默认加载最新）"},
                    },
                },
            },
        ]

    async def run(self) -> None:
        """启动MCP服务器，监听stdin"""
        self._running = True
        logger.info("MCP Server started, listening on stdin...")

        # 初始化引擎
        await self.engine.init()

        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                message = json.loads(line.strip())
                response = await self._handle_message(message)
                self._send_response(response)

            except json.JSONDecodeError as e:
                self._send_error(-32700, f"Parse error: {e}")
            except Exception as e:
                self._send_error(-32603, f"Internal error: {e}")

    async def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条消息"""
        msg_type = message.get("type", "")
        msg_id = message.get("id")

        if msg_type == "initialize":
            return {
                "type": "initialized",
                "id": msg_id,
                "capabilities": {
                    "tools": self.get_tool_definitions(),
                },
            }
        elif msg_type == "tools/list":
            return {
                "type": "tools/list/response",
                "id": msg_id,
                "tools": self.get_tool_definitions(),
            }
        elif msg_type == "tools/call":
            tool_name = message.get("name", "")
            arguments = message.get("arguments", {})

            if tool_name in self._tool_handlers:
                result = await self._tool_handlers[tool_name](arguments)
                return {
                    "type": "tools/call/response",
                    "id": msg_id,
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                }
            else:
                return {
                    "type": "tools/call/response",
                    "id": msg_id,
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                }
        else:
            return {
                "type": "error",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {msg_type}"},
            }

    def _send_response(self, response: Dict[str, Any]) -> None:
        """发送响应到stdout"""
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _send_error(self, code: int, message: str) -> None:
        """发送错误响应"""
        self._send_response({
            "type": "error",
            "error": {"code": code, "message": message},
        })

    # ---- 工具处理器 ----

    async def _handle_skill_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        skill = self.engine.create_skill(
            name=args["name"],
            description=args.get("description", ""),
        )
        return {"skill_id": skill.id, "name": skill.name}

    async def _handle_skill_execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.engine.execute(
            skill_id=args["skill_id"],
            context=args.get("context", {}),
            success=args.get("success", True),
            value=args.get("value", 0.0),
        )
        return result

    async def _handle_skill_analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        embedding = args.get("embedding", [])
        return self.engine.analyze(embedding)

    async def _handle_skill_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        skills = self.engine.list_skills()
        return {"skills": skills, "count": len(skills)}

    async def _handle_skill_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.stats()

    async def _handle_skill_save(self, args: Dict[str, Any]) -> Dict[str, Any]:
        filepath = self.storage.save(label=args.get("label", ""))
        return {"saved_to": filepath}

    async def _handle_skill_load(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.engine = self.storage.load(filepath=args.get("filepath"))
        self.storage.bind_engine(self.engine)
        return {"status": "loaded", "skills_count": len(self.engine._skills)}


async def main():
    """MCP服务器入口点"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = MCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
