/**
 * types.ts - 共享类型定义
 *
 * self-evolving-skill TypeScript SDK 的类型定义文件。
 */

/** Skill 状态枚举 */
export enum SkillStatus {
  CREATED = 'created',
  ACTIVE = 'active',
  EVOLVING = 'evolving',
  STABLE = 'stable',
  DEPRECATED = 'deprecated',
}

/** Skill 数据结构 */
export interface Skill {
  id: string;
  name: string;
  description: string;
  status: string;
  value_score?: number;
  evolution_count?: number;
  abstraction_level?: string;
}

/** MCP 工具定义结构 */
export interface MCPToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: string;
    properties: Record<string, unknown>;
    required?: string[];
  };
}

/** 残差金字塔分解结果 */
export interface DecompositionResult {
  residual_ratio: number;
  novelty_score: number;
  suggested_abstraction: string;
  coverage: number;
}

/** 触发判断结果 */
export interface TriggerResult {
  triggered: boolean;
  energy_ratio: number;
  threshold: number;
  trigger_rate: number;
  reason: string;
}

/** 执行结果 */
export interface ExecutionResult {
  skill_id: string;
  success: boolean;
  value: number;
  decomposition: DecompositionResult;
  trigger: TriggerResult;
  evolution?: Record<string, unknown>;
}
