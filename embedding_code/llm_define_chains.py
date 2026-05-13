#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成产业链环节的文本描述定义

该脚本读取产业链路径文件，调用LLM API为每个产业链环节生成精确的文本定义，
并将结果保存为JSON文件。支持失败重试机制和进度跟踪。
同时支持生成主产业链名称的文本定义。
"""

import json
import time
import sys
import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
import yaml

# Import prompts from definition_prompt module
from prompt.definition_prompt import get_active_prompts, build_user_prompt


def load_config():
    """
    从配置文件加载脚本的默认参数配置

    Returns:
        dict: 包含 llm_define_chains 脚本配置的字典，如果加载失败则返回空字典
    """
    config_file = Path(__file__).parent / 'config.yaml'

    if not config_file.exists():
        print(f"警告: 配置文件不存在: {config_file}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config and 'llm_define_chains' in config:
                return config['llm_define_chains']
            else:
                print(f"警告: 配置文件中未找到 llm_define_chains 配置")
                return {}
    except (yaml.YAMLError, IOError) as e:
        print(f"警告: 无法读取配置文件: {str(e)}")
        return {}


def sanitize_industry_name(industry_name):
    """
    Convert industry name to a safe filename component.
    
    Args:
        industry_name: Industry name in Chinese or other characters
        
    Returns:
        str: Sanitized filename-safe string
    """
    # Remove special characters and replace spaces with underscores
    # Keep only alphanumeric, Chinese characters, and underscores
    sanitized = re.sub(r'[^\w\u4e00-\u9fff]+', '_', industry_name)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Convert to lowercase if it's ASCII
    if sanitized.isascii():
        sanitized = sanitized.lower()
    return sanitized


def load_existing_output(output_file):
    """
    Load existing output file to check for previous results and failed paths.
    
    Args:
        output_file: Path to the output JSON file
        
    Returns:
        dict: Dictionary with 'results' and 'failed_paths' keys, or None if file doesn't exist
    """
    output_path = Path(output_file)
    if not output_path.exists():
        return None
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                'results': data.get('results', []),
                'failed_paths': data.get('failed_paths', [])
            }
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告: 无法读取现有输出文件: {str(e)}")
        return None


def merge_results(old_results, new_results, remaining_failed_paths):
    """
    Merge old and new results, updating the failed paths list.
    
    Args:
        old_results: List of previous successful results
        new_results: List of newly generated results
        remaining_failed_paths: List of paths that still failed after retry
        
    Returns:
        dict: Merged output data with 'results' and 'failed_paths' keys
    """
    merged_data = {
        'results': old_results + new_results,
        'failed_paths': remaining_failed_paths
    }
    return merged_data


def call_llm_api(user_prompt, system_prompt, api_url, model_name, temperature, 
                 max_tokens, max_retries, retry_delay):
    """
    Call the LLM API with retry logic.
    
    Args:
        user_prompt: The user prompt text
        system_prompt: The system prompt text
        api_url: LLM API endpoint URL
        model_name: Model name to use
        temperature: Temperature parameter for generation
        max_tokens: Maximum output tokens
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        str: The LLM response text, or None if all retries failed
    """
    payload = {
        "model": model_name,
        "contents": [
            {
                "parts": [
                    {
                        "text": user_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        },
        "systemInstruction": {
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=600
            )
            response.raise_for_status()
            
            # Extract the text from the response
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return text
            
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                print(f"\n  Error on attempt {attempt + 1}: {str(e)}")
                print(f"  Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print(f"\n  Failed after {max_retries} attempts: {str(e)}")
                return None
    
    return None


def generate_main_chain_definition(industry_name, all_chains_text, output_file,
                                   api_url, model_name, temperature, max_tokens,
                                   max_retries, retry_delay):
    """
    生成主产业链名称的文本定义
    
    Args:
        industry_name: 主产业链名称
        all_chains_text: 完整的产业链图谱文本
        output_file: 输出的JSON文件路径
        api_url: LLM API端点URL
        model_name: 模型名称
        temperature: 生成温度参数
        max_tokens: 最大输出token数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        tuple: (success_count, failed_count)
    """
    print()
    print("=" * 70)
    print("主产业链定义生成器")
    print("=" * 70)
    print()
    print(f"正在为主产业链 '{industry_name}' 生成定义...")
    
    # Get prompt templates
    prompts = get_active_prompts()
    
    # Format prompts
    system_prompt = prompts['main_chain_system_prompt']
    user_prompt = prompts['main_chain_user_prompt'].format(
        industry_name=industry_name,
        all_chains_text=all_chains_text
    )
    
    # Call LLM API
    response_text = call_llm_api(
        user_prompt, system_prompt, api_url, model_name,
        temperature, max_tokens, max_retries, retry_delay
    )
    
    # Prepare output
    if response_text:
        output_data = {
            "industry_name": industry_name,
            "result": response_text,
            "status": "success"
        }
        success = 1
        failed = 0
    else:
        output_data = {
            "industry_name": industry_name,
            "result": None,
            "status": "failed"
        }
        success = 0
        failed = 1
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save result
    print(f"正在保存主产业链定义到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("主产业链定义生成总结")
    print("=" * 70)
    print(f"主产业链名称: {industry_name}")
    print(f"生成状态: {'成功' if success else '失败'}")
    print(f"结果已保存到: {output_file}")
    print("=" * 70)
    print()
    
    return success, failed


def generate_definitions(input_file, output_file, api_url, model_name, 
                        temperature, max_tokens, max_retries, retry_delay,
                        generate_main_chain):
    """
    生成产业链环节定义的主函数
    
    Args:
        input_file: 输入的产业链路径文件
        output_file: 输出的JSON文件（细分环节定义）
        api_url: LLM API端点URL
        model_name: 模型名称
        temperature: 生成温度参数
        max_tokens: 最大输出token数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        generate_main_chain: 是否生成主产业链定义
        
    Returns:
        tuple: (success_count, failed_count) for sub-chain definitions
    """
    print("=" * 70)
    print("产业链环节定义生成器")
    print("=" * 70)
    print()
    
    # Always read from Excel file to get complete chain context
    print(f"正在读取产业链路径文件: {input_file}")
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # Read Excel file with sheet name 'for_cal'
    df = pd.read_excel(input_file, sheet_name='for_cal')
    
    # Extract industry name from the first row
    industry_name = str(df.iloc[0]['chain_name']) if pd.notna(df.iloc[0]['chain_name']) else '未命名产业'
    print(f"检测到主产业链名称: {industry_name}")
    
    # Concatenate columns with "——" separator to form all chain paths
    all_lines = []
    for _, row in df.iterrows():
        parts = [
            str(row['chain_name']) if pd.notna(row['chain_name']) else '',
            str(row['chain_level_1_name']) if pd.notna(row['chain_level_1_name']) else '',
            str(row['chain_level_2_name']) if pd.notna(row['chain_level_2_name']) else '',
            str(row['chain_level_3_name']) if pd.notna(row['chain_level_3_name']) else '',
            str(row['chain_level_4_name']) if pd.notna(row['chain_level_4_name']) else ''
        ]
        # Filter out empty strings and join with "——"
        parts = [p for p in parts if p]
        if parts:
            all_lines.append("——".join(parts))
    
    # Join all lines to create complete chain context for system prompt
    all_chains_text = "\n".join(all_lines)
    
    # Get prompt templates
    prompts = get_active_prompts()
    
    # Format the system_prompt with industry_name and all_chains_text
    formatted_system_prompt = prompts['system_prompt'].format(
        industry_name=industry_name,
        all_chains_text=all_chains_text
    )
    
    # Generate main chain definition if requested
    if generate_main_chain:
        # Construct main chain output filename
        output_path = Path(output_file)
        main_chain_output = output_path.parent / f"{output_path.stem}_main{output_path.suffix}"
        
        generate_main_chain_definition(
            industry_name=industry_name,
            all_chains_text=all_chains_text,
            output_file=str(main_chain_output),
            api_url=api_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    # Check for existing output and failed paths (resume mode)
    existing_data = load_existing_output(output_file)
    resume_mode = False
    old_results = []
    
    if existing_data and existing_data['failed_paths']:
        resume_mode = True
        old_results = existing_data['results']
        failed_paths_to_retry = existing_data['failed_paths']
        
        print()
        print("检测到现有输出文件，进入续跑模式")
        print(f"已有成功结果: {len(old_results)} 条")
        print(f"待重试失败路径: {len(failed_paths_to_retry)} 条")
        print()
        
        # Use only failed paths for processing
        lines_to_process = failed_paths_to_retry
        
    elif existing_data and not existing_data['failed_paths']:
        print()
        print("现有输出文件中没有失败路径，无需重试")
        print(f"已有成功结果: {len(existing_data['results'])} 条")
        print("=" * 70)
        return len(existing_data['results']), 0
        
    else:
        # Normal mode: process all paths
        lines_to_process = all_lines
    
    # Parse chain paths to process
    chain_paths = []
    for line in lines_to_process:
        parts = line.split("——")
        l0 = parts[0]  # Main industry chain (first element)
        full_path = line  # Complete path
        chain_paths.append((l0, full_path))
    
    if resume_mode:
        print(f"正在重试 {len(chain_paths)} 条失败路径")
    else:
        print(f"找到 {len(chain_paths)} 条产业链路径待处理")
    print()
    
    # Process each chain path
    results = []
    failed_paths = []
    
    print("正在生成细分环节定义...")
    for l0, full_path in tqdm(chain_paths, desc="处理进度", unit="条"):
        # Build the user prompt using the imported function
        user_prompt = build_user_prompt(l0, full_path)
        
        # Call the LLM API with formatted system prompt
        response_text = call_llm_api(
            user_prompt, formatted_system_prompt, api_url, model_name, 
            temperature, max_tokens, max_retries, retry_delay
        )
        
        if response_text:
            results.append(response_text)
        else:
            failed_paths.append(full_path)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge results if in resume mode
    if resume_mode:
        output_data = merge_results(old_results, results, failed_paths)
    else:
        output_data = {
            "results": results,
            "failed_paths": failed_paths
        }
    
    print()
    print(f"正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("细分环节定义生成总结")
    print("=" * 70)
    
    if resume_mode:
        print(f"原有成功结果: {len(old_results)} 条")
        print(f"本次处理路径: {len(chain_paths)} 条")
        print(f"本次成功: {len(results)} 条")
        print(f"本次失败: {len(failed_paths)} 条")
        print(f"---")
        print(f"累计成功总数: {len(output_data['results'])} 条")
        print(f"剩余失败总数: {len(output_data['failed_paths'])} 条")
    else:
        print(f"总路径数: {len(chain_paths)}")
        print(f"成功处理: {len(results)}")
        print(f"失败数量: {len(failed_paths)}")
    
    if failed_paths:
        print()
        print("失败的路径:")
        for path in failed_paths:
            print(f"  - {path}")
    
    print()
    print(f"结果已保存到: {output_file}")
    print("=" * 70)
    
    return len(output_data['results']), len(output_data['failed_paths'])


def main():
    """主函数"""
    # 加载配置文件中的默认值
    config = load_config()

    parser = argparse.ArgumentParser(
        description='生成产业链环节的文本描述定义',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_file',
        type=str,
        default=config.get('input_file', 'sample_data/sample_industry_chain.xlsx'),
        help='输入的产业链路径文件路径'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=config.get('output_file', None),
        help='输出的JSON文件路径（细分环节定义）。如果不指定，将根据产业名称自动生成'
    )

    parser.add_argument(
        '--api_url',
        type=str,
        default=config.get('api_url', 'http://localhost:8000/v1'),
        help='LLM API端点URL'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default=config.get('model_name', 'your-llm-model'),
        help='使用的模型名称'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=config.get('temperature', 0),
        help='生成温度参数'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=config.get('max_tokens', 10240),
        help='最大输出token数'
    )

    parser.add_argument(
        '--max_retries',
        type=int,
        default=config.get('max_retries', 3),
        help='失败后的最大重试次数'
    )

    parser.add_argument(
        '--retry_delay',
        type=int,
        default=config.get('retry_delay', 10),
        help='重试之间的延迟时间（秒）'
    )

    parser.add_argument(
        '--generate_main_chain',
        type=bool,
        default=config.get('generate_main_chain', True),
        help='是否生成主产业链定义'
    )
    
    args = parser.parse_args()
    
    # If output_file is not specified, generate it based on industry name
    if args.output_file is None:
        # Read the industry name from Excel
        df = pd.read_excel(args.input_file, sheet_name='for_cal', nrows=1)
        industry_name = str(df.iloc[0]['chain_name']) if pd.notna(df.iloc[0]['chain_name']) else '未命名产业'
        sanitized_name = sanitize_industry_name(industry_name)
        script_dir = Path(__file__).parent
        args.output_file = str(script_dir / 'chain_definitions_llm_output' / f'llm_chain_definitions_{sanitized_name}.json')
        print(f"自动生成输出文件名: {args.output_file}")
    
    # 执行定义生成
    generate_definitions(
        input_file=args.input_file,
        output_file=args.output_file,
        api_url=args.api_url,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        generate_main_chain=args.generate_main_chain
    )


if __name__ == "__main__":
    main()
