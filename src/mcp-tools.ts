/**
 * mcp-tools.ts - MCP 工具定义
 *
 * 为 MCP 协议（Model Context Protocol）定义 self-evolving-skill 的所有工具。
 * 这些工具定义可以直接注册到 MCP 客户端（如 Claude Desktop）。
 *
 * 支持的 MCP 工具：
 * - skill_create    : 创建Skill
 * - skill_execute   : 执行并学习
 * - skill_analyze   : 分析嵌入
 * - skill_list      : 列出Skills
 * - skill_stats     : 系统统计
 * - skill_save      : 持久化保存
 * - skill_load      : 加载
 */

import type { MCPToolDefinition } from './types';

/**
 * 获取所有 MCP 工具定义
 *
 * 这些定义符合 MCP 协议规范，
 * 可以直接用于 MCP 服务器的 tools/list 响应。
 */
export function getSelfEvolvingSkillTools(): MCPToolDefinition[] {
  return [
    {
      name: 'skill_create',
      description: '创建一个新的自我演化Skill。Skill将自动进入学习循环，基于执行结果和价值反馈进行演化。',
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Skill名称，建议使用描述性名称如 "DataAnalyzer" 或 "ResponseFormatter"',
          },
          description: {
            type: 'string',
            description: 'Skill的详细描述，说明其功能和适用场景',
          },
        },
        required: ['name'],
      },
    },
    {
      name: 'skill_execute',
      description:
        '执行指定Skill并触发元认知学习循环。系统会自动进行残差金字塔分解、自适应反思触发、经验回放和价值门控评估。',
      inputSchema: {
        type: 'object',
        properties: {
          skill_id: {
            type: 'string',
            description: '目标Skill的ID（可通过skill_list获取）',
          },
          context: {
            type: 'object',
            description: '执行上下文，包含任务相关的键值对数据',
          },
          success: {
            type: 'boolean',
            description: '执行是否成功，影响价值评估',
            default: true,
          },
          value: {
            type: 'number',
            description: '执行价值评分 (0.0~1.0)，越高表示越有价值',
            default: 0.0,
          },
        },
        required: ['skill_id', 'context'],
      },
    },
    {
      name: 'skill_analyze',
      description:
        '分析嵌入向量，返回残差金字塔分解结果：残差比率、新颖性评分、建议的抽象层级。',
      inputSchema: {
        type: 'object',
        properties: {
          embedding: {
            type: 'array',
            items: { type: 'number' },
            description: '待分析的高维嵌入向量，通常来自文本/特征的向量化表示',
          },
        },
        required: ['embedding'],
      },
    },
    {
      name: 'skill_list',
      description: '列出系统中所有已创建的Skills及其当前状态、价值评分和演化次数。',
      inputSchema: {
        type: 'object',
        properties: {},
      },
    },
    {
      name: 'skill_stats',
      description:
        '获取系统的全面统计信息，包括：Skill数量分布、残差金字塔状态、触发器统计、经验回放缓存统计、价值门控统计。',
      inputSchema: {
        type: 'object',
        properties: {},
      },
    },
    {
      name: 'skill_save',
      description: '将当前系统状态持久化保存到本地存储。支持带标签的版本管理，自动保留最近10个版本。',
      inputSchema: {
        type: 'object',
        properties: {
          label: {
            type: 'string',
            description: '保存标签，便于辨识不同版本（如 "before_refactor", "after_training"）',
          },
        },
      },
    },
    {
      name: 'skill_load',
      description: '从本地存储加载已保存的系统状态。可以指定文件路径，默认加载最新版本。',
      inputSchema: {
        type: 'object',
        properties: {
          filepath: {
            type: 'string',
            description: '状态文件路径（可选），不指定则加载最新保存的版本',
          },
        },
      },
    },
  ];
}

/**
 * 将工具定义注册到 MCP 服务器的辅助函数
 *
 * @param registerTool 注册工具的回调函数
 */
export function registerTools(
  registerTool: (tool: MCPToolDefinition) => void
): void {
  const tools = getSelfEvolvingSkillTools();
  for (const tool of tools) {
    registerTool(tool);
  }
}

/**
 * 快速启动工具注册
 *
 * 适用于 MCP 服务器实现中的快速集成。
 * 示例：
 * ```typescript
 * import { quickRegister } from './mcp-tools';
 * quickRegister((tool) => mcpServer.addTool(tool));
 * ```
 */
export const quickRegister = registerTools;
