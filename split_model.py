# -*- coding: utf-8 -*-
"""
将大型模型文件分割成小块
"""

import torch
import os
import pickle

def split_model_file(model_path, chunk_size_mb=25):
    """
    将PyTorch模型文件分割成多个小文件
    
    Args:
        model_path: 原始模型文件路径
        chunk_size_mb: 每个块的大小（MB）
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # 加载原始模型
    print(f"加载模型: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 序列化整个state_dict
    serialized_data = pickle.dumps(state_dict)
    total_size = len(serialized_data)
    
    print(f"模型总大小: {total_size / (1024*1024):.2f} MB")
    
    # 计算需要的块数
    num_chunks = (total_size + chunk_size_bytes - 1) // chunk_size_bytes
    print(f"将分割为 {num_chunks} 个块")
    
    # 分割并保存
    base_name = os.path.splitext(model_path)[0]
    
    for i in range(num_chunks):
        start_idx = i * chunk_size_bytes
        end_idx = min((i + 1) * chunk_size_bytes, total_size)
        
        chunk_data = serialized_data[start_idx:end_idx]
        chunk_filename = f"{base_name}_chunk_{i:03d}.pkl"
        
        with open(chunk_filename, 'wb') as f:
            f.write(chunk_data)
        
        chunk_size_mb_actual = len(chunk_data) / (1024*1024)
        print(f"保存块 {i+1}/{num_chunks}: {chunk_filename} ({chunk_size_mb_actual:.2f} MB)")
    
    # 保存元数据
    metadata = {
        'num_chunks': num_chunks,
        'total_size': total_size,
        'chunk_size_bytes': chunk_size_bytes,
        'base_name': base_name
    }
    
    metadata_filename = f"{base_name}_metadata.json"
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"元数据保存到: {metadata_filename}")
    print("分割完成！")

def load_chunked_model(base_name):
    """
    从分块文件中重新加载模型
    
    Args:
        base_name: 模型文件的基础名称（不含扩展名）
    
    Returns:
        state_dict: 重新组装的模型状态字典
    """
    import json
    
    # 加载元数据
    metadata_filename = f"{base_name}_metadata.json"
    with open(metadata_filename, 'r') as f:
        metadata = json.load(f)
    
    num_chunks = metadata['num_chunks']
    total_size = metadata['total_size']
    
    print(f"加载 {num_chunks} 个模型块...")
    
    # 重新组装数据
    serialized_data = bytearray(total_size)
    chunk_size_bytes = metadata['chunk_size_bytes']
    
    for i in range(num_chunks):
        chunk_filename = f"{base_name}_chunk_{i:03d}.pkl"
        
        with open(chunk_filename, 'rb') as f:
            chunk_data = f.read()
        
        start_idx = i * chunk_size_bytes
        end_idx = start_idx + len(chunk_data)
        serialized_data[start_idx:end_idx] = chunk_data
        
        print(f"加载块 {i+1}/{num_chunks}: {chunk_filename}")
    
    # 反序列化
    state_dict = pickle.loads(bytes(serialized_data))
    print("模型重新组装完成！")
    
    return state_dict

if __name__ == "__main__":
    # 分割模型文件
    model_file = "epoch50_acc_0.88048_pr_0.85800_roc_0.94890.pth"
    
    if os.path.exists(model_file):
        split_model_file(model_file, chunk_size_mb=25)
    else:
        print(f"模型文件不存在: {model_file}")
        
    # 测试重新加载
    print("\n测试重新加载...")
    base_name = os.path.splitext(model_file)[0]
    try:
        reloaded_state_dict = load_chunked_model(base_name)
        print("重新加载测试成功！")
    except Exception as e:
        print(f"重新加载测试失败: {e}")
