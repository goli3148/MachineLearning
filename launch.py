import os
import threading
from threading import Thread
# config
ignores = ['.venv', 'launch.py']
python_file_name_ex = ".py"

paths = []
names = []

def similarity(filename):
    for igonre in ignores:
        if igonre in str(filename):
            return True

def walk():
    for root, dirs, files in os.walk('.'):
        if not similarity(root):
            for name in files:
                if python_file_name_ex in name and not similarity(name):
                    path = f"{root}\\{name}"
                    if not path in paths: 
                        paths.append(path)
                        names.append(name.replace(".py",""))
    print("\n  Menu:")
    for index in range(len(paths)):
        print(f"\t{index+1} : {names[index]}")
    print("  Settings")
    print("\t-1 : reset the list")
    print("\t-2 : exit")
    print("\t-3 : clear screen")

def lauch_program():
    while True:
        walk()
        print("  Choose index from list above(integer only):", end=" ")
        try: 
            index = int(input())
            if index == -1: pass
            elif index == -2: exit()
            elif index == -3: os.system("cls")
            else:
                print("EXECUTION ------------------------------------------------------------------- \n")
                th = Thread(target=os.system, args=(f'py {paths[index-1]}',))
                th.start()
                th.join()
                print("\nEND OF EXECUTION -------------------------------------------------------------------")
                os.system('pause')
        except Exception as e:
            print(f"erroe occourred : {e}")

lauch_program()
