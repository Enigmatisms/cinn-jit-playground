""" Compare two nsys profiling results
and generate a formatted xlsx table
"""

import pandas as pd

def parse_mem_stats_file(file_path: str) -> list[tuple[str, float]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            
            try:
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError("Format error")
                
                name, value_str = parts
                value = float(value_str.replace('%', '').strip())
                data.append((name, value))
                
            except Exception as e:
                print(f"[Mem stats file {file_path}] skipping line {line_num}: {str(e)}")
    return data

def parse_nsys_stats_file(file_path: str) -> list[tuple[float, float, float, str, str]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # skip three lines of headers
        [next(f) for _ in range(3)]
        
        for line_num, line in enumerate(f, 4):  # start from line 4
            stripped = line.strip()
            if not stripped:
                continue
            
            try:
                parts = stripped.split('\t')
                if len(parts) != 5:
                    raise ValueError("Not enough columns")
                
                # process comma-separated values
                cum_time = float(parts[0].replace(',', ''))
                total_time = float(parts[1].replace(',', ''))
                avg_time = float(parts[2].replace(',', ''))
                reduce_flag = parts[3].strip()
                kernel_name = parts[4].strip()
                
                data.append((cum_time, total_time, avg_time, reduce_flag, kernel_name))
                
            except Exception as e:
                print(f"[Nsys stats file {file_path}] skipping line {line_num}: {str(e)}")
    return data

def merge_data(mem1_path: str, mem2_path: str, nsys1_path: str, nsys2_path: str, output_path: str) -> None:
    mem1_data = parse_mem_stats_file(mem1_path)
    mem2_data = parse_mem_stats_file(mem2_path)
    nsys1_data = parse_nsys_stats_file(nsys1_path)
    nsys2_data = parse_nsys_stats_file(nsys2_path)
    
    lengths = [len(mem1_data), len(mem2_data), len(nsys1_data), len(nsys2_data)]
    if len(set(lengths)) != 1:
        print(f"Error: mismatching row counts (A1={lengths[0]}, A2={lengths[1]}, B1={lengths[2]}, B2={lengths[3]})")
        return
    
    dict1 = {}
    dict2 = {}
    for i in range(len(mem1_data)):
        a1_name, a1_val = mem1_data[i]
        a2_name, a2_val = mem2_data[i]
        
        b1_cum, b1_total, b1_avg, b1_reduce, b1_name = nsys1_data[i]
        
        b2_cum, b2_total, b2_avg, b2_reduce, b2_name = nsys2_data[i]
        
        dict1[a1_name] = [a1_val, b1_cum, b1_total, b1_avg, b1_reduce]
        dict2[a2_name] = [a2_val, b2_cum, b2_total, b2_avg, b2_reduce]
    
    merged = []
    for key in dict1.keys():
        if not key in dict2:
            continue
        
        a1_val, b1_cum, b1_total, b1_avg, b1_reduce = dict1[key]
        a2_val, b2_cum, b2_total, b2_avg, b2_reduce = dict2[key]
        
        merged.append({
            'Name': key,
            
            'NHWC bandwidth(%)': a1_val,
            'TFG bandwidth(%)': a2_val,
            
            'NHWC Avg Time': b1_avg,
            'TFG Avg Time': b2_avg,
            'BW Improvement(%)': (a1_val - a2_val) / a2_val * 100,
            'RT Improvement(%)': (b1_avg - b2_avg) / b2_avg * 100,
            
            'NHWC Cum Time': b1_cum,
            'TFG Cum Time': b2_cum,
            'NHWC Total Time': b1_total,
            'TFG Total Time': b2_total,
        })
    
    df = pd.DataFrame(merged)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        worksheet = writer.sheets['Sheet1']
        
        worksheet.column_dimensions['A'].width = 50
        for i in range(10):
            worksheet.column_dimensions[chr(ord('B') + i)].width = 22
        
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    merge_data(
        mem1_path="mem_result_5.log",
        mem2_path="mem_result_4.log",
        nsys1_path="nsys_stats_5.txt",
        nsys2_path="nsys_stats_4.txt",
        output_path="mbnetv3_comparison.xlsx"
    )