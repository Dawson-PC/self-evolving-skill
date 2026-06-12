/**
 * index.ts - TypeScript SDK 主入口
 *
 * Self-Evolving Skill 的 TypeScript 端 SDK，
 * 提供 SelfEvolvingSkillEngine 类和核心工具函数。
 *
 * 使用方式:
 * ```typescript
 * import { SelfEvolvingSkillEngine } from 'self-evolving-skill';
 *
 * const engine = new SelfEvolvingSkillEngine();
 * await engine.init();
 *
 * const { skillId } = await engine.createSkill({ name: 'Analyzer' });
 * const stats = await engine.stats();
 * ```
 */

import { Skill, SkillStatus, MCPToolDefinition } from './types';

/**
 * SelfEvolvingSkillEngine - 自我演化Skill引擎
 *
 * 通过 MCP 协议与后端 Python 核心通信，
 * 提供完整的 Skill 生命周期管理。
 */
export class SelfEvolvingSkillEngine {
  private mcpEndpoint: string;
  private initialized: boolean = false;
  private requestId: number = 0;

  constructor(mcpEndpoint: string = 'http://localhost:8080') {
    this.mcpEndpoint = mcpEndpoint;
  }

  /**
   * 初始化引擎（建立与MCP服务器的连接）
   */
  async init(): Promise<void> {
    // 实际场景中在此建立 MCP 连接
    // 当前为 TypeScript SDK 网关层实现
    this.initialized = true;
  }

  /**
   * 创建新Skill
   */
  async createSkill(params: { name: string; description?: string }): Promise<{ skillId: string }> {
    this.checkInit();
    const result = await this.callMCP('skill_create', params);
    return { skillId: result.skill_id };
  }

  /**
   * 执行Skill并触发学习
   */
  async execute(params: {
    skillId: string;
    context: Record<string, unknown>;
    success?: boolean;
    value?: number;
  }): Promise<Record<string, unknown>> {
    this.checkInit();
    return this.callMCP('skill_execute', {
      skill_id: params.skillId,
      context: params.context,
      success: params.success ?? true,
      value: params.value ?? 0.0,
    });
  }

  /**
   * 分析嵌入向量
   */
  async analyze(embedding: number[]): Promise<Record<string, unknown>> {
    this.checkInit();
    return this.callMCP('skill_analyze', { embedding });
  }

  /**
   * 列出所有Skills
   */
  async listSkills(): Promise<Skill[]> {
    this.checkInit();
    const result = await this.callMCP('skill_list', {});
    return result.skills as Skill[];
  }

  /**
   * 获取系统统计
   */
  async stats(): Promise<Record<string, unknown>> {
    this.checkInit();
    return this.callMCP('skill_stats', {});
  }

  /**
   * 持久化保存
   */
  async save(label?: string): Promise<string> {
    this.checkInit();
    const result = await this.callMCP('skill_save', { label: label ?? '' });
    return result.saved_to as string;
  }

  /**
   * 加载已保存状态
   */
  async load(filepath?: string): Promise<void> {
    this.checkInit();
    await this.callMCP('skill_load', { filepath: filepath ?? '' });
  }

  /**
   * 获取MCP工具定义
   */
  getToolDefinitions(): MCPToolDefinition[] {
    return [
      {
        name: 'skill_create',
        description: '创建新的Skill',
        inputSchema: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Skill名称' },
            description: { type: 'string', description: '描述' },
          },
          required: ['name'],
        },
      },
      {
        name: 'skill_execute',
        description: '执行Skill并触发学习',
        inputSchema: {
          type: 'object',
          properties: {
            skill_id: { type: 'string', description: 'Skill ID' },
            context: { type: 'object', description: '执行上下文' },
            success: { type: 'boolean', description: '是否成功' },
            value: { type: 'number', description: '价值评分' },
          },
          required: ['skill_id', 'context'],
        },
      },
      {
        name: 'skill_analyze',
        description: '分析嵌入向量',
        inputSchema: {
          type: 'object',
          properties: {
            embedding: {
              type: 'array',
              items: { type: 'number' },
              description: '嵌入向量',
            },
          },
          required: ['embedding'],
        },
      },
      {
        name: 'skill_list',
        description: '列出所有Skills',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'skill_stats',
        description: '系统统计信息',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'skill_save',
        description: '持久化保存当前状态',
        inputSchema: {
          type: 'object',
          properties: {
            label: { type: 'string', description: '保存标签' },
          },
        },
      },
      {
        name: 'skill_load',
        description: '加载已保存的状态',
        inputSchema: {
          type: 'object',
          properties: {
            filepath: { type: 'string', description: '文件路径（可选）' },
          },
        },
      },
    ];
  }

  private checkInit(): void {
    if (!this.initialized) {
      throw new Error('Engine not initialized. Call init() first.');
    }
  }

  private async callMCP(
    toolName: string,
    args: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    this.requestId++;

    // 实际 MCP 协议调用
    // 当前为占位实现，通过 HTTP 转发到 Python MCP 服务器
    const response = await fetch(`${this.mcpEndpoint}/mcp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'tools/call',
        id: this.requestId,
        name: toolName,
        arguments: args,
      }),
    });

    if (!response.ok) {
      throw new Error(`MCP call failed: ${response.statusText}`);
    }

    const data = await response.json();
    if (data.isError) {
      throw new Error(`MCP tool error: ${data.content?.[0]?.text ?? 'Unknown error'}`);
    }

    const content = data.content?.[0]?.text;
    return content ? JSON.parse(content) : {};
  }
}

/**
 * 便捷函数：创建引擎实例并初始化
 */
export async function createEngine(
  mcpEndpoint?: string
): Promise<SelfEvolvingSkillEngine> {
  const engine = new SelfEvolvingSkillEngine(mcpEndpoint);
  await engine.init();
  return engine;
}
