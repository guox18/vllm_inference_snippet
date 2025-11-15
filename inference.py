#!/usr/bin/env python3
"""
vLLM 推理脚本 - 简洁易用的推理框架

使用方法:
    python inference.py \\
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
        --input_file data/input.jsonl \\
        --output_file data/output.jsonl \\
        --input_parser examples/input_parser.py \\
        --input_parser_fn parse_to_messages \\
        --output_parser examples/output_parser.py \\
        --output_parser_fn process_output \\
        --num_gpus 4 \\
        --tensor_parallel_size 2
"""

import argparse
import json
import importlib.util
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Callable
from tqdm import tqdm

from vllm import LLM, SamplingParams


def load_function_from_file(file_path: str, function_name: str) -> Callable:
    """从指定的 Python 文件中加载函数"""
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def load_data(input_file: str) -> List[Dict[str, Any]]:
    """自动检测并加载 JSON 或 JSONL 格式的数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 尝试解析为 JSON 数组
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"✓ 检测到 JSON 格式，加载 {len(data)} 条数据")
            return data
    except json.JSONDecodeError:
        pass
    
    # 按 JSONL 格式解析
    data = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠ 跳过无效行: {line[:50]}... 错误: {e}")
    
    print(f"✓ 检测到 JSONL 格式，加载 {len(data)} 条数据")
    return data


def save_output(output_file: str, item: Dict[str, Any], mode: str = 'a'):
    """以 JSONL 格式追加保存单条数据"""
    with open(output_file, mode, encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_checkpoint(checkpoint_file: str) -> int:
    """加载检查点，返回已处理的数据条数"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(checkpoint_file: str, processed_count: int):
    """保存检查点"""
    with open(checkpoint_file, 'w') as f:
        f.write(str(processed_count))


def run_inference(args):
    """主推理流程"""
    
    # 1. 加载输入解析函数和输出解析函数
    print("\n" + "="*60)
    print("📥 加载自定义解析函数...")
    input_parser = load_function_from_file(args.input_parser, args.input_parser_fn)
    output_parser = load_function_from_file(args.output_parser, args.output_parser_fn)
    print(f"✓ 输入解析函数: {args.input_parser}::{args.input_parser_fn}")
    print(f"✓ 输出解析函数: {args.output_parser}::{args.output_parser_fn}")
    
    # 2. 加载输入数据
    print("\n" + "="*60)
    print("📂 加载输入数据...")
    all_data = load_data(args.input_file)

    # 3. 处理数据分片（支持手动分片和 torchrun 分片）
    if args.shard_id is not None and args.num_shards is not None:
        # 手动分片模式
        shard_id = args.shard_id
        num_shards = args.num_shards
        rank = shard_id
        world_size = num_shards
    else:
        # torchrun 模式（从环境变量读取）
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # 数据并行：每个进程处理一部分数据
        shard_size = len(all_data) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else len(all_data)
        data = all_data[start_idx:end_idx]
        print(
            f"✓ 数据分片 {rank}/{world_size}，处理数据 [{start_idx}:{end_idx}]（共 {len(data)} 条）"
        )
    else:
        data = all_data
        print(f"✓ 单进程模式，处理全部 {len(data)} 条数据")

    # 4. 设置输出文件（多分片时添加后缀）
    # 如果明确指定了分片参数，或者 world_size > 1，都添加分片后缀
    if world_size > 1 or (args.shard_id is not None and args.num_shards is not None):
        # 为每个分片创建独立的输出文件
        base_name = args.output_file.rsplit(".", 1)
        if len(base_name) == 2:
            output_file = f"{base_name[0]}_shard{rank}.{base_name[1]}"
        else:
            output_file = f"{args.output_file}_shard{rank}"
        print(f"✓ 输出文件: {output_file}")
    else:
        output_file = args.output_file

    # 5. 断点续传
    checkpoint_file = f"{output_file}.checkpoint"
    if args.resume:
        processed_count = load_checkpoint(checkpoint_file)
        if processed_count > 0:
            print(f"✓ 从检查点恢复，跳过前 {processed_count} 条数据")
            data = data[processed_count:]
    else:
        processed_count = 0
        # 清空输出文件
        open(output_file, "w").close()
    
    if len(data) == 0:
        print("✓ 所有数据已处理完成")
        return

    # 5. 设置 GPU 设备（手动分片模式下根据 shard_id 分配 GPU）
    if args.shard_id is not None and args.num_shards is not None:
        # 手动分片模式：根据 shard_id 自动分配 GPU
        gpu_start = args.shard_id * args.tensor_parallel_size
        gpu_ids = list(range(gpu_start, gpu_start + args.tensor_parallel_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"✓ 分片 {args.shard_id} 使用 GPU: {gpu_ids}")
        actual_tp_size = args.tensor_parallel_size
    elif world_size > 1:
        # torchrun 模式：从环境变量获取 rank 并分配 GPU
        gpu_start = rank * args.tensor_parallel_size
        gpu_ids = list(range(gpu_start, gpu_start + args.tensor_parallel_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"\n✓ Rank {rank} 使用 GPU: {gpu_ids}")
        actual_tp_size = args.tensor_parallel_size
    else:
        actual_tp_size = args.tensor_parallel_size

    # 6. 初始化 vLLM 模型
    print("\n" + "=" * 60)
    print("🚀 初始化 vLLM 模型...")
    print(f"   模型: {args.model_name_or_path}")
    print(f"   张量并行: {actual_tp_size}")
    print(f"   GPU 内存利用率: {args.gpu_memory_utilization}")

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=actual_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
    )

    tokenizer = llm.get_tokenizer()

    # 7. 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    print(f"✓ 采样参数: temperature={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}, max_tokens={args.max_tokens}")

    # 8. 批量推理
    print("\n" + "=" * 60)
    print("⚙️  开始推理...")

    batch_size = args.batch_size
    total_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(0, len(data), batch_size), desc="推理进度", total=total_batches
    ):
        batch_data = data[batch_idx : batch_idx + batch_size]

        # 解析输入数据为 messages
        prompts = []
        for item in batch_data:
            try:
                messages = input_parser(item)
                # 使用 tokenizer 的 chat template
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
            except Exception as e:
                print(f"\n⚠ 输入解析失败: {e}")
                prompts.append(None)

        # 过滤掉解析失败的数据
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        valid_data = [batch_data[i] for i in valid_indices]

        if not valid_prompts:
            continue

        # 执行推理
        outputs = llm.generate(valid_prompts, sampling_params)

        # 处理输出
        for item, output in zip(valid_data, outputs):
            response = output.outputs[0].text
            try:
                result_item = output_parser(item, response)
                save_output(output_file, result_item)
            except Exception as e:
                print(f"\n⚠ 输出解析失败: {e}")

        # 更新检查点
        if args.resume:
            processed_count += len(batch_data)
            save_checkpoint(checkpoint_file, processed_count)

    # 9. 清理检查点文件
    if args.resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("\n" + "=" * 60)
    print("✅ 推理完成！")
    print(f"📁 输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="vLLM 推理脚本")

    # 必需参数
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="模型名称或路径"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="输入数据文件 (JSON 或 JSONL)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="输出数据文件 (JSONL)"
    )
    parser.add_argument(
        "--input_parser",
        type=str,
        required=True,
        help="输入解析函数所在的 Python 文件路径",
    )
    parser.add_argument(
        "--input_parser_fn", type=str, required=True, help="输入解析函数名称"
    )
    parser.add_argument(
        "--output_parser",
        type=str,
        required=True,
        help="输出解析函数所在的 Python 文件路径",
    )
    parser.add_argument(
        "--output_parser_fn", type=str, required=True, help="输出解析函数名称"
    )

    # 分布式参数
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="使用的 GPU 总数（用于数据并行计算）"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="张量并行大小（TP）"
    )

    # 手动数据分片参数
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="当前分片的 ID（0开始），用于手动数据并行",
    )
    parser.add_argument(
        "--num_shards", type=int, default=None, help="总分片数，用于手动数据并行"
    )

    # 模型参数
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU 内存利用率")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="是否信任远程代码")
    parser.add_argument("--max_model_len", type=int, default=0,
                        help="最大模型长度（0 表示使用默认值）")
    
    # 采样参数
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="top-p 采样")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="top-k 采样（-1 表示不使用）")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成 token 数")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--resume", action="store_true",
                        help="是否从检查点恢复")
    
    args = parser.parse_args()

    # 参数验证
    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard_id 和 --num_shards 必须同时指定或同时不指定")

    if args.shard_id is not None:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(f"--shard_id 必须在 [0, {args.num_shards - 1}] 范围内")
        print(f"\n🔹 手动数据分片模式")
        print(f"   当前分片: {args.shard_id}/{args.num_shards}")
        print(f"   张量并行: {args.tensor_parallel_size}")
        print(
            f"   使用 GPU: {list(range(args.shard_id * args.tensor_parallel_size, (args.shard_id + 1) * args.tensor_parallel_size))}"
        )
    else:
        # 检查是否使用 torchrun
        data_parallel = args.num_gpus // args.tensor_parallel_size
        if data_parallel > 1:
            print(f"\n⚠ 注意: 检测到数据并行度为 {data_parallel}")
            print(f"   请使用 torchrun 启动脚本:")
            print(f"   torchrun --nproc_per_node={data_parallel} inference.py ...")
            print(f"   或使用手动分片模式: --shard_id 0 --num_shards {data_parallel}")
            print(f"   当前将以单进程模式运行")
    
    run_inference(args)


if __name__ == "__main__":
    main()

