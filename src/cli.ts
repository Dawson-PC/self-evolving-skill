/**
 * cli.ts - CLI 命令行接口
 *
 * 提供命令行方式使用 self-evolving-skill 的功能。
 * 支持以下命令：
 * - list     : 列出所有Skill
 * - create   : 创建Skill
 * - execute  : 执行并学习
 * - analyze  : 分析嵌入
 * - stats    : 系统统计
 * - save     : 持久化保存
 * - load     : 加载
 *
 * 使用方式:
 * ```bash
 * node dist/cli.js list
 * node dist/cli.js create --name "MySkill"
 * node dist/cli.js execute <id> --success
 * node dist/cli.js analyze --embedding '[0.1,0.2,...]'
 * node dist/cli.js stats
 * ```
 */

import { SelfEvolvingSkillEngine } from './index';

// ---- 类型定义 ----

interface CommandHandler {
  (engine: SelfEvolvingSkillEngine, args: string[]): Promise<void>;
}

interface Command {
  name: string;
  description: string;
  usage: string;
  handler: CommandHandler;
}

// ---- 命令处理器 ----

const commands: Record<string, Command> = {
  list: {
    name: 'list',
    description: '列出所有Skills',
    usage: 'list',
    handler: async (engine) => {
      const skills = await engine.listSkills();
      if (skills.length === 0) {
        console.log('No skills found.');
        return;
      }
      console.log('Skills:');
      console.log('─'.repeat(60));
      for (const skill of skills) {
        console.log(
          `  ${skill.id.padEnd(8)} | ${skill.name.padEnd(20)} | ${skill.status.padEnd(10)} | value: ${skill.value_score?.toFixed(2) ?? '0.00'}`
        );
      }
      console.log('─'.repeat(60));
      console.log(`Total: ${skills.length} skills`);
    },
  },

  create: {
    name: 'create',
    description: '创建新Skill',
    usage: 'create --name <name> [--description <desc>]',
    handler: async (engine, args) => {
      const name = parseArg(args, '--name');
      const description = parseArg(args, '--description') ?? '';

      if (!name) {
        console.error('Error: --name is required');
        console.log(`Usage: ${commands.create.usage}`);
        process.exit(1);
      }

      const result = await engine.createSkill({ name, description });
      console.log(`Skill created: ${result.skillId} (${name})`);
    },
  },

  execute: {
    name: 'execute',
    description: '执行Skill并学习',
    usage: 'execute <skill_id> [--context \'{"key":"value"}\'] [--success true] [--value 0.5]',
    handler: async (engine, args) => {
      const skillId = args[0];
      if (!skillId) {
        console.error('Error: skill_id is required');
        console.log(`Usage: ${commands.execute.usage}`);
        process.exit(1);
      }

      const contextStr = parseArg(args, '--context');
      const context = contextStr ? JSON.parse(contextStr) : {};
      const success = parseArg(args, '--success') !== 'false';
      const value = parseFloat(parseArg(args, '--value') ?? '0.0');

      const result = await engine.execute({
        skillId,
        context,
        success,
        value,
      });

      console.log(`Execute result:`);
      console.log(JSON.stringify(result, null, 2));
    },
  },

  analyze: {
    name: 'analyze',
    description: '分析嵌入向量',
    usage: 'analyze --embedding \'[0.1, 0.2, ...]\'',
    handler: async (engine, args) => {
      const embeddingStr = parseArg(args, '--embedding');
      if (!embeddingStr) {
        console.error('Error: --embedding is required');
        console.log(`Usage: ${commands.analyze.usage}`);
        process.exit(1);
      }

      const embedding = JSON.parse(embeddingStr);
      const result = await engine.analyze(embedding);

      console.log('Analysis result:');
      console.log(JSON.stringify(result, null, 2));
    },
  },

  stats: {
    name: 'stats',
    description: '系统统计信息',
    usage: 'stats',
    handler: async (engine) => {
      const stats = await engine.stats();
      console.log('System statistics:');
      console.log(JSON.stringify(stats, null, 2));
    },
  },

  save: {
    name: 'save',
    description: '持久化保存当前状态',
    usage: 'save [--label <label>]',
    handler: async (engine, args) => {
      const label = parseArg(args, '--label');
      const savedTo = await engine.save(label);
      console.log(`State saved to: ${savedTo}`);
    },
  },

  load: {
    name: 'load',
    description: '加载已保存的状态',
    usage: 'load [--filepath <path>]',
    handler: async (engine, args) => {
      const filepath = parseArg(args, '--filepath');
      await engine.load(filepath);
      console.log('State loaded successfully.');
    },
  },

  help: {
    name: 'help',
    description: '显示帮助信息',
    usage: 'help [command]',
    handler: async (_engine, args) => {
      const cmdName = args[0];
      if (cmdName && commands[cmdName]) {
        const cmd = commands[cmdName];
        console.log(`\n  ${cmd.name} - ${cmd.description}`);
        console.log(`  Usage: ${cmd.usage}\n`);
        return;
      }
      console.log('\n  Self-Evolving Skill CLI');
      console.log('  Usage: cli <command> [options]\n');
      console.log('  Commands:');
      for (const cmd of Object.values(commands)) {
        console.log(`    ${cmd.name.padEnd(12)} ${cmd.description}`);
      }
      console.log('');
    },
  },
};

// ---- 辅助函数 ----

function parseArg(args: string[], key: string): string | undefined {
  const index = args.indexOf(key);
  if (index !== -1 && index + 1 < args.length) {
    return args[index + 1];
  }
  return undefined;
}

// ---- 入口 ----

async function main() {
  const [, , command, ...restArgs] = process.argv;

  if (!command || command === 'help') {
    await commands.help.handler(null as unknown as SelfEvolvingSkillEngine, restArgs);
    process.exit(0);
  }

  if (!commands[command]) {
    console.error(`Unknown command: ${command}`);
    console.log('Run "help" for usage information.');
    process.exit(1);
  }

  try {
    const engine = await createEngine();
    await commands[command].handler(engine, restArgs);
    process.exit(0);
  } catch (error) {
    console.error('Error:', (error as Error).message);
    process.exit(1);
  }
}

main();
