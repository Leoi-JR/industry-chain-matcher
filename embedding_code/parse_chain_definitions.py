#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain Definitions Formatter

This script processes LLM-generated chain definition JSON files and generates:
1. A human-readable markdown file with organized definitions
2. A structured JSON list for further processing

Author: Auto-generated
Date: 2025
"""

import json
import re
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import yaml


def sanitize_industry_name(industry_name: str) -> str:
    sanitized = re.sub(r'[^\w一-鿿]+', '_', industry_name)
    sanitized = sanitized.strip('_')
    if sanitized.isascii():
        sanitized = sanitized.lower()
    return sanitized


def load_config():
    """
    从配置文件加载脚本的默认参数配置

    Returns:
        dict: 包含 parse_chain_definitions 脚本配置的字典，如果加载失败则返回空字典
    """
    config_file = Path(__file__).parent / 'config.yaml'

    if not config_file.exists():
        print(f"警告: 配置文件不存在: {config_file}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config and 'parse_chain_definitions' in config:
                return config['parse_chain_definitions']
            else:
                print(f"警告: 配置文件中未找到 parse_chain_definitions 配置")
                return {}
    except (yaml.YAMLError, IOError) as e:
        print(f"警告: 无法读取配置文件: {str(e)}")
        return {}


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_json_string(json_str: str) -> str:
    """
    Remove markdown code block markers from JSON string.
    
    Args:
        json_str: JSON string potentially wrapped in markdown code blocks
        
    Returns:
        Cleaned JSON string
    """
    # Remove ```json\n prefix
    if json_str.startswith("```json\n"):
        json_str = json_str[8:]
    elif json_str.startswith("```json"):
        json_str = json_str[7:]
    
    # Remove trailing ```
    if json_str.endswith("\n```"):
        json_str = json_str[:-4]
    elif json_str.endswith("```"):
        json_str = json_str[:-3]
    
    return json_str.strip()


def extract_main_industry_data(main_file_path: str) -> Tuple[str, str]:
    """
    Extract industry name and definition from main industry JSON file.
    
    Args:
        main_file_path: Path to the main industry JSON file
        
    Returns:
        Tuple of (industry_name, definition)
        
    Raises:
        KeyError: If expected fields are missing
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        data = load_json_file(main_file_path)
        
        # Extract industry_name
        if "industry_name" not in data:
            raise KeyError(f"'industry_name' field not found in {main_file_path}")
        industry_name = data["industry_name"]
        
        # Extract and clean result
        if "result" not in data:
            raise KeyError(f"'result' field not found in {main_file_path}")
        result_str = data["result"]
        
        # Clean and parse the result JSON string
        cleaned_result = clean_json_string(result_str)
        result_data = json.loads(cleaned_result)
        
        # Extract definition
        if "definition" not in result_data:
            raise KeyError(f"'definition' field not found in parsed result from {main_file_path}")
        definition = result_data["definition"]
        
        return industry_name, definition
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON in {main_file_path}: {str(e)}",
            e.doc, e.pos
        )


def extract_chain_definitions(chain_file_path: str) -> List[Dict[str, str]]:
    """
    Extract chain path definitions from chain definitions JSON file.
    
    Args:
        chain_file_path: Path to the chain definitions JSON file
        
    Returns:
        List of dictionaries with chain_path, dependency_type, and definition
        
    Raises:
        KeyError: If expected fields are missing
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        data = load_json_file(chain_file_path)
        
        # Extract results array
        if "results" not in data:
            raise KeyError(f"'results' field not found in {chain_file_path}")
        results = data["results"]
        
        if not isinstance(results, list):
            raise ValueError(f"'results' field in {chain_file_path} is not a list")
        
        chain_definitions = []
        
        for idx, result_str in enumerate(results):
            try:
                # Clean and parse each result
                cleaned_result = clean_json_string(result_str)
                result_data = json.loads(cleaned_result)
                
                # Extract required fields
                required_fields = ["chain_path", "dependency_type", "embedding_friendly_definition"]
                for field in required_fields:
                    if field not in result_data:
                        print(f"Warning: '{field}' missing in result #{idx+1}, skipping...", 
                              file=sys.stderr)
                        continue
                
                # Add to list if all fields present
                if all(field in result_data for field in required_fields):
                    chain_definitions.append({
                        "chain_path": result_data["chain_path"],
                        "dependency_type": result_data["dependency_type"],
                        "definition": result_data["embedding_friendly_definition"]
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse result #{idx+1}: {str(e)}", file=sys.stderr)
                continue
        
        return chain_definitions
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON in {chain_file_path}: {str(e)}",
            e.doc, e.pos
        )


def generate_markdown(industry_name: str, main_definition: str, 
                     chain_definitions: List[Dict[str, str]]) -> str:
    """
    Generate markdown content from definitions.
    
    Args:
        industry_name: Name of the industry
        main_definition: Definition of the main industry
        chain_definitions: List of chain path definitions
        
    Returns:
        Markdown formatted string
    """
    lines = []
    
    # Header
    lines.append(f"# {industry_name} 产业链定义\n")
    
    # Main industry section
    lines.append("## 主产业链\n")
    lines.append(f"**产业链名称**: {industry_name}\n")
    lines.append(f"**定义**: {main_definition}\n")
    lines.append("---\n")
    
    # Chain definitions section
    lines.append("## 产业链环节定义\n")
    
    for chain_def in chain_definitions:
        lines.append(f"### {chain_def['chain_path']}\n")
        lines.append(f"**类型**: {chain_def['dependency_type']}\n")
        lines.append(f"**定义**: {chain_def['definition']}\n")
    
    return "\n".join(lines)


def generate_json_list(industry_name: str, main_definition: str,
                      chain_definitions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generate structured JSON list from definitions.
    
    Args:
        industry_name: Name of the industry
        main_definition: Definition of the main industry
        chain_definitions: List of chain path definitions
        
    Returns:
        List of dictionaries with standardized fields
    """
    result_list = []
    
    # Add main industry record first
    result_list.append({
        "name": industry_name,
        "type": "主产业链名称",
        "definition": main_definition,
        "dependency_type": ""
    })
    
    # Add all chain path records
    for chain_def in chain_definitions:
        result_list.append({
            "name": chain_def["chain_path"],
            "type": "产业链环节链路",
            "definition": chain_def["definition"],
            "dependency_type": chain_def["dependency_type"]
        })
    
    return result_list


def save_markdown(content: str, output_path: str) -> None:
    """
    Save markdown content to file.
    
    Args:
        content: Markdown content
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Markdown file saved: {output_path}")


def save_json(data: List[Dict[str, str]], output_path: str) -> None:
    """
    Save JSON data to file.
    
    Args:
        data: Data to save
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON file saved: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # 加载配置文件中的默认值
    config = load_config()

    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    # 处理 input_dir 和 output_dir 的默认值
    default_input_dir = str(script_dir / config.get('input_dir', 'chain_definitions_llm_output'))
    default_output_dir = str(script_dir / config.get('output_dir', 'chain_definitions_formatted'))

    parser = argparse.ArgumentParser(
        description="Process chain definition JSON files and generate markdown and JSON outputs"
    )

    parser.add_argument(
        "--industry_name",
        type=str,
        default=config.get('industry_name', '示例产业链'),
        help="Industry name (default: 示例产业链)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=default_input_dir,
        help="Input directory containing JSON files (default: chain_definitions_llm_output)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Output directory for generated files (default: chain_definitions_formatted)"
    )

    return parser.parse_args()

def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Construct file paths (sanitize to match llm_define_chains.py output filenames)
    sanitized_name = sanitize_industry_name(args.industry_name)
    main_file_path = os.path.join(
        args.input_dir,
        f"llm_chain_definitions_{sanitized_name}_main.json"
    )
    chain_file_path = os.path.join(
        args.input_dir,
        f"llm_chain_definitions_{sanitized_name}.json"
    )
    
    markdown_output_path = os.path.join(
        args.output_dir,
        f"chain_definitions_{args.industry_name}.md"
    )
    json_output_path = os.path.join(
        args.output_dir,
        f"chain_definitions_{args.industry_name}_list.json"
    )
    
    try:
        print(f"Processing industry: {args.industry_name}")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        # Extract main industry data
        print(f"Reading main industry file: {main_file_path}")
        industry_name, main_definition = extract_main_industry_data(main_file_path)
        print(f"  ✓ Extracted main industry definition")
        
        # Extract chain definitions
        print(f"Reading chain definitions file: {chain_file_path}")
        chain_definitions = extract_chain_definitions(chain_file_path)
        print(f"  ✓ Extracted {len(chain_definitions)} chain definitions")
        print()
        
        # Generate markdown
        print("Generating markdown file...")
        markdown_content = generate_markdown(industry_name, main_definition, chain_definitions)
        save_markdown(markdown_content, markdown_output_path)
        
        # Generate JSON list
        print("Generating JSON list file...")
        json_list = generate_json_list(industry_name, main_definition, chain_definitions)
        save_json(json_list, json_output_path)
        
        print()
        print("=" * 60)
        print("Processing completed successfully!")
        print(f"Total records: {len(json_list)} (1 main industry + {len(chain_definitions)} chain paths)")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected field - {str(e)}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

