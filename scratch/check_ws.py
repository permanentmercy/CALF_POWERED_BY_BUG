with open(r'f:\DOWNLOAD\参考文献-大模型\CALF-main\CALF-main\models\CALF.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(227, 234):
        print(f"{i+1}: {repr(lines[i])}")
