# 最基础版本
def function_01(matrix):
    res_array=[]
    for array in matrix:
        for a in array:
            if a%3==0:
                continue
            print(a)
            if a%2==0:
                pass
            else:
                a*=2
            res_array.append(a)
    return res_array

# 优化版本
def function_01_optimize(matrix):
    return [
      a*2 if a%2!=0 else a
      for array in matrix
      for a in array
      if a %3 !=0
    ]

#未优化
def function_02(authors_books):
    books_authors={}
    for author in authors_books.keys():
        books=authors_books[author]
        for book in books:
            books_authors.setdefault(book,[]).append(author)
    return books_authors

#优化版本
def function_02_optimize(authors_books):
    books_authors = {}
    for author, books in authors_books.items():
        for book in books:
            books_authors.setdefault(book, []).append(author)
    return books_authors

#基础版本
def function_03():
    num_set=[]
    for i in range(3,100) :
        if i+2 <100:
            num_set.append([i,i+2]) if is_prime(i) and is_prime(i+2) else num_set
    return num_set

#确定是否质数        
def is_prime(i):
    for j in range(2,i) :
       if(i%j==0):
           return False
    return True 

#优化版本
def function_03_optimize():
    return [
        [i,i+2]
        for i in range(3,100)
        if is_prime(i) and is_prime(i+2)
        ]


#基础版本
def function_04(nested_config):
    return flatten_dict(nested_config,'')

def flatten_dict(nested_config,prefix=''):
    item={}
    for k,v in nested_config.items():
        new_key=prefix + '.' + k if prefix else k
        if(isinstance(v,dict)):
            item.update(flatten_dict(v,new_key))
        else:
            item[new_key]=v
    return item

#优化版本
def function_04_optimize(nested_config,prefix=''):
    return {
        prefix+"."+k if prefix else k:v
        for key,value in nested_config.items()
        for k,v in (
            function_04_optimize(v,prefix+"."+k if prefix else k).items
            if isinstance(value,dict)
            else {key:value}.items()
        )
    }


#基础版本&优化版本
def function_05(scores_long):
    student_dic={}
    for name,subject,score in scores_long:
        student_dic.setdefault(name,{}).update({subject : score})
    return student_dic

import re
#基础版
def function_07(text=''):
    # 去除符号，转小写，分词
    # Python 中的 r 表示“原始字符串”，防止反斜杠 \ 被当作转义字符。
    words = re.sub(r'[^\w\s]', '', text).lower().split()
    
    # 过滤以 p 开头，且以 l 或 n 结尾的词
    return {
        word
        for word in words
        if word.startswith('p') and (word.endswith('l') or word.endswith('n'))
    }

def function_08(dict_chess):
    return {
        str(k)+str(v):
            'white' 
                if (ord(k.lower()) -ord('a') + v-1)%2==0
                else 'black'
        for k,v in dict_chess.items()
    }

def function_9(max_num):
    return {
        count:[
            i 
            for i in range(21)
            if(bin(i).count('1')==count) 
        ]
        for count in range(0,bin(max_num).count('1')+1)
    }
from datetime import datetime

def function_10():
    data_dict = {}  # 存储日记：{id: {id, timestamp, content, status}}
    default_value_data_id = 0  # 初始ID从0开始，第一次add会变为1
    # 可以添加操作历史栈用于undo/redo
    history = []  # 存储操作历史：[(操作类型, 数据)]
    
    while True:
        user_command = input("""
            ===============================================================
            add <内容>: 创建一条新日记，status 初始为 'active'。
            delete <id>: 将指定 id 的日记条目的 status 修改为 'deleted'。
            list: 显示所有 status 为 'active' 的日记。
            search <关键词>: 在所有 active 日记的 content 中搜索关键词。
            undo: 撤销上一次的 add 或 delete 操作。
            redo: 恢复上一次被撤销的操作。
            export <文件名.txt>: 导出所有 active 日记到文件。
            exit: 退出程序。
            请输入命令：
            ===============================================================\n
        """).strip()  # 去除首尾空格
        
        # 分割命令前缀和参数（处理无参数的情况）
        parts = user_command.split(' ', 1)
        command_prifix = parts[0] if parts else ''
        command_data = parts[1] if len(parts) > 1 else ''
        
        match command_prifix:
            case 'add':
                if not command_data:  # 检查内容是否为空
                    print("错误：添加的日记内容不能为空")
                    continue
                # 记录操作历史（用于undo）
                history.append(('add', default_value_data_id + 1))
                # 添加新日记
                default_value_data_id += 1
                data_dict[default_value_data_id] = {  # 直接赋值比update更简洁
                    'id': default_value_data_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'content': command_data,
                    'status': 'active'
                }
                print(f'日记添加成功（ID: {default_value_data_id}）\n内容：{command_data}')
            
            case 'delete':
                if not command_data:
                    print("错误：请指定要删除的日记ID")
                    continue
                try:
                    target_id = int(command_data)
                except ValueError:
                    print("错误：ID必须是数字")
                    continue
                
                temp_data = data_dict.get(target_id)
                if temp_data and temp_data['status'] == 'active':  # 只删除active的
                    # 记录操作历史（存储删除前的状态）
                    history.append(('delete', target_id, temp_data['status']))
                    temp_data['status'] = 'deleted'
                    print(f'日记ID: {target_id} 已标记为删除\n内容：{temp_data["content"]}')
                elif temp_data and temp_data['status'] == 'deleted':
                    print(f'日记ID: {target_id} 已处于删除状态')
                else:
                    print(f'错误：未找到ID为 {target_id} 的日记')
            
            case 'list':
                active_diaries = [v for v in data_dict.values() if v['status'] == 'active']
                if not active_diaries:
                    print("暂无活跃日记")
                else:
                    print(f"共 {len(active_diaries)} 条活跃日记：")
                    for diary in active_diaries:
                        print(f"\nID: {diary['id']}")
                        print(f"时间: {diary['timestamp']}")
                        print(f"内容: {diary['content']}")
                        print('-' * 50)
            
            case 'search':
                if not command_data:
                    print("错误：请输入搜索关键词")
                    continue
                # 搜索active日记中包含关键词的内容
                matched = [
                    v for v in data_dict.values()
                    if v['status'] == 'active' and command_data in v['content']
                ]
                if not matched:
                    print(f"未找到包含 '{command_data}' 的活跃日记")
                else:
                    print(f"找到 {len(matched)} 条匹配结果：")
                    for diary in matched:
                        print(f"\nID: {diary['id']}（{diary['timestamp']}）")
                        print(f"内容: {diary['content']}")
                        print('-' * 50)
            
            case 'undo':
                # 简单实现：撤销最后一次add/delete（需完善历史记录逻辑）
                if not history:
                    print("没有可撤销的操作")
                    continue
                last_op = history.pop()
                if last_op[0] == 'add':
                    # 撤销add：删除最后添加的日记
                    added_id = last_op[1]
                    if added_id in data_dict:
                        del data_dict[added_id]
                        default_value_data_id -= 1  # 恢复ID计数
                        print(f"已撤销添加操作（ID: {added_id}）")
                elif last_op[0] == 'delete':
                    # 撤销delete：恢复状态为active
                    deleted_id = last_op[1]
                    if deleted_id in data_dict:
                        data_dict[deleted_id]['status'] = 'active'
                        print(f"已撤销删除操作（ID: {deleted_id}）")
            
            case 'redo':
                # 需维护一个redo栈，暂未实现
                print("redo功能暂未实现")
            
            case 'export':
                if not command_data:
                    print("错误：请指定导出文件名（如 export diary.txt）")
                    continue
                try:
                    active_diaries = [v for v in data_dict.values() if v['status'] == 'active']
                    with open(command_data, 'w', encoding='utf-8') as f:
                        for diary in active_diaries:
                            f.write(f"ID: {diary['id']}\n")
                            f.write(f"时间: {diary['timestamp']}\n")
                            f.write(f"内容: {diary['content']}\n")
                            f.write('-' * 50 + '\n')
                    print(f"成功导出 {len(active_diaries)} 条活跃日记到 {command_data}")
                except Exception as e:
                    print(f"导出失败：{str(e)}")
            
            case 'exit':
                print("退出程序")
                break
            
            case '':  # 处理空输入
                continue
            
            case _:
                print(f"错误：未知命令 '{command_prifix}'，请重新输入")

# 运行函数
if __name__ == "__main__":
    function_10()
