import subprocess, re, ast, json, pickle
import pandas as pd

list_ret_all = []
count = 0
for i in range(1, 5):
    for j in range(1, 12):
        cmd = 'python run_search.py -p ' + str(i) + ' -s ' + str(j)
        res = subprocess.check_output(cmd, shell=True)
        res = res.decode('utf8')
        ret = re.sub('(.*?)(\\()(.*?)(\\))(\s+)', "", res, flags=re.MULTILINE)
        ret = re.sub('(Solving\s+)', "], [", ret, flags=re.MULTILINE)
        ret = re.sub('(\.\.\.)', "", ret, flags=re.MULTILINE)
        ret = re.sub('(using)', ",", ret, flags=re.MULTILINE)
        ret = re.sub('(# Actions   Expansions   Goal Tests   New Nodes)', ",", ret, flags=re.MULTILINE)
        ret = re.sub('(Plan length:)', ",", ret, flags=re.MULTILINE)
        ret = re.sub('(Time elapsed in seconds:)', ",", ret, flags=re.MULTILINE)
        ret = re.sub('(Time elapsed in seconds:)', ",", ret, flags=re.MULTILINE)
        pattern = re.compile('([0-9]{1})(\s+)([0-9]{1})', flags=re.MULTILINE)
        ret = re.sub(pattern, '\\1' + "," + "\\3", ret)
        ret = re.sub('\r', "", ret, flags=re.MULTILINE)
        ret = ret.replace("], ", "[", 1)
        ret = ret + "]]"
        ret = re.sub('(\s+,\s+)', ", ", ret, flags=re.MULTILINE)
        ret = re.sub('(,\s+)', ", ", ret, flags=re.MULTILINE)
        pattern = re.compile('([0-9]{1})(,)([0-9]{1})', flags=re.MULTILINE)
        ret = re.sub(pattern, '\\1' + ", " + "\\3", ret)
        ret = re.sub('\\[', "['", ret, flags=re.MULTILINE)
        ret = re.sub("\\]", "']", ret, flags=re.MULTILINE)
        ret = re.sub(", ", "', '", ret, flags=re.MULTILINE)
        ret = re.sub("\\['\\[", "[[", ret, flags=re.MULTILINE)
        ret = re.sub("\\]'\\]", "]]", ret, flags=re.MULTILINE)
        ret = re.sub('\\n', "", ret, flags=re.MULTILINE)
        ret = re.sub("'\\]'", "']", ret, flags=re.MULTILINE)
        ret = re.sub("'\\['", "['", ret, flags=re.MULTILINE)
        list_ret = ast.literal_eval(ret)
        list_ret_all += list_ret
        count += 1
        try:
            print(count, i, j)
            with open('list_ret_all.pkl', 'wb') as f:
                pickle.dump(list_ret_all, f)
        except:
            print("save failed")

list_for_df = [tuple(x) for x in list_ret_all]
my_index = pd.Index([i for i in range(1, len(list_for_df) + 1)], name='Configuration No.')
df = pd.DataFrame(data=list_for_df, index=my_index, columns=['Problem', 'Search Function', 'Actions', 'Expansions', 'Goal Tests', 'New Nodes', 'Plan Length', 'Time Elapsed in Seocnds'])

df.to_excel("result.xlsx")

"""

Solving Air Cargo Problem 1 using breadth_first_search...

# Actions   Expansions   Goal Tests   New Nodes
    20          43          56         178    

Plan length: 6  Time elapsed in seconds: 0.002403599999999992
Solving Air Cargo Problem 1 using depth_first_graph_search...

# Actions   Expansions   Goal Tests   New Nodes
    20          21          22          84    

Plan length: 20  Time elapsed in seconds: 0.0013861000000000012
Solving Air Cargo Problem 2 using breadth_first_search...

# Actions   Expansions   Goal Tests   New Nodes
    72         3343        4609       30503   

Plan length: 9  Time elapsed in seconds: 0.7638887
Solving Air Cargo Problem 2 using depth_first_graph_search...

# Actions   Expansions   Goal Tests   New Nodes
    72         624         625         5602   

Plan length: 619  Time elapsed in seconds: 1.0484716

"""